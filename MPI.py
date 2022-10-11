import torch
import torch.nn as nn
import torch.nn.functional as torchf
import os
import imageio
import time
import cv2
from utils import *
from NeRF_modules import get_embedder
from utils_mpi import *
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    rasterize_meshes,
    RasterizationSettings,
    TexturesUV,
    Textures
)

ACTIVATES = {'relu': torch.relu,
             'sigmoid': torch.sigmoid,
             'unsigmoid': lambda x: torch.log(x.clamp(1e-6, 1 - 1e-6) / (1 - x.clamp(1e-6, 1 - 1e-6))),
             'exp': torch.exp,
             'none': lambda x: x,
             'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
             'softplus': lambda x: nn.Softplus()(x - 1),
             'tanh': torch.tanh,
             'clamp': lambda x: torch.clamp(x, 0, 1),
             'clamp_g': lambda x: x + (torch.clamp(x, 0, 1) - x).detach(),
             'plus05': lambda x: x + 0.5}

ALPHA_INIT_VAL = -3.


class MPMesh(nn.Module):
    def __init__(self, args, H, W, ref_extrin, ref_intrin, near, far):
        super(MPMesh, self).__init__()
        self.args = args
        self.upsample_stage = args.upsample_stage
        mpi_h, mpi_w = int(args.mpi_h_scale * H), int(args.mpi_w_scale * W)
        self.mpi_d, self.near, self.far = args.mpi_d, near, far
        self.mpi_h_verts, self.mpi_w_verts = args.mpi_h_verts, args.mpi_w_verts
        self.H, self.W = H, W
        self.atlas_grid_h, self.atlas_grid_w = args.atlas_grid_h, self.mpi_d // args.atlas_grid_h
        assert self.mpi_d % self.atlas_grid_h == 0, "mpi_d and atlas_grid_h should match"
        self.is_sparse = False
        self.has_dyn = False
        self.atlas_full_h = int(self.atlas_grid_h * mpi_h)
        self.atlas_full_w = int(self.atlas_grid_w * mpi_w)

        assert ref_extrin.shape == (4, 4) and ref_intrin.shape == (3, 3)
        self.register_buffer("ref_extrin", torch.tensor(ref_extrin))
        self.register_buffer("ref_intrin", torch.tensor(ref_intrin).float())

        # construct the vertices
        planedepth = make_depths(self.mpi_d, near, far).float().flip(0)
        self.register_buffer("planedepth", planedepth)

        # get intrin for mapping entire MPI to image, in order to generate vertices
        self.H_start, self.W_start = (mpi_h - H) // 2, (mpi_w - W) // 2
        ref_intrin_mpi = get_new_intrin(self.ref_intrin, - self.H_start, - self.W_start)

        # generate primitive vertices
        # #############################
        verts = gen_mpi_vertices(mpi_h, mpi_w, ref_intrin_mpi,
                                 args.mpi_h_verts, args.mpi_w_verts, planedepth)
        if args.normalize_verts:
            scaling = self.planedepth
            verts = (verts.reshape(len(scaling), -1) / scaling[:, None]).reshape_as(verts)

        # generate faces
        # ########################
        verts_indice = torch.arange(len(verts)).reshape(self.mpi_d, args.mpi_h_verts, args.mpi_w_verts)
        faces013 = torch.stack([verts_indice[:, :-1, :-1], verts_indice[:, :-1, 1:], verts_indice[:, 1:, 1:]], -1)
        faces320 = torch.stack([verts_indice[:, 1:, 1:], verts_indice[:, 1:, :-1], verts_indice[:, :-1, :-1]], -1)
        faces = torch.cat([faces013.reshape(-1, 1, 3), faces320.reshape(-1, 1, 3)], dim=1).reshape(-1, 3)

        # generate uv coordinate
        # ########################
        uvs_plane = torch.meshgrid([torch.arange(self.atlas_grid_h) / self.atlas_grid_h,
                                    torch.arange(self.atlas_grid_w) / self.atlas_grid_w])
        uvs_plane = torch.stack(uvs_plane[::-1], dim=-1) * 2 - 1
        uvs_voxel_size = (- uvs_plane[-1, -1] + 1).reshape(1, 1, 2)
        uvs_voxel = torch.meshgrid([torch.linspace(0, 1, args.mpi_h_verts), torch.linspace(0, 1, args.mpi_w_verts)])
        uvs_voxel = torch.stack(uvs_voxel[::-1], dim=-1).reshape(1, -1, 2) * uvs_voxel_size
        uvs = (uvs_plane.reshape(-1, 1, 2) + uvs_voxel.reshape(1, -1, 2)).reshape(-1, 2)

        self.register_buffer("uvfaces", faces.clone().long())
        self._verts = nn.Parameter(verts, requires_grad=True)
        self.register_buffer("faces", faces.long())
        self.optimize_geometry = False
        self.register_parameter("uvs", nn.Parameter(uvs, requires_grad=True))

        # configure and initializing the atlas
        self.view_embed_fn, self.view_cnl = get_embedder(args.multires_views, input_dim=3)
        self.rgb_mlp_type = args.rgb_mlp_type
        if args.rgb_mlp_type == "direct":
            self.feat2rgba = lambda x: x[..., :4]
            atlas_cnl = 4
            atlas = torch.rand((1, atlas_cnl, int(self.atlas_full_h), int(self.atlas_full_w)))
            atlas[:, -1] = ALPHA_INIT_VAL
            self.use_viewdirs = False
        elif args.rgb_mlp_type == "rgb_sh":
            atlas_cnl = 3 * 4 + 1  # one for alpha, 9 for base
            atlas = torch.rand((1, atlas_cnl, int(self.atlas_full_h), int(self.atlas_full_w)))
            atlas[:, 0] = ALPHA_INIT_VAL
            self.feat2rgba = SphericalHarmoic_RGB(atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
        else:
            raise RuntimeError(f"rgbmlp_type = {args.rgb_mlp_type} not recognized")

        self.register_parameter("atlas", nn.Parameter(atlas, requires_grad=True))
        if args.learn_loop_mask:
            atlas_mask = torch.ones_like(atlas[:, :1]) * ALPHA_INIT_VAL
            self.register_parameter("atlas_mask", nn.Parameter(atlas_mask, requires_grad=True))

        self.rgb_activate = ACTIVATES[args.rgb_activate]
        self.alpha_activate = ACTIVATES[args.alpha_activate]

    def get_optimizer(self):
        args = self.args
        base_lr = args.lrate
        verts_lr = args.lrate * args.optimize_verts_gain

        all_params = {k: v for k, v in self.named_parameters()}
        verts_params_list = ["_verts"]
        base_params_list = set(all_params.keys()) - set(verts_params_list)
        params = [
            {'params': [all_params[k] for k in base_params_list]},  # param_group 0
            {'params': [all_params[k] for k in verts_params_list],  # param_group 1
             'lr': verts_lr}
        ]
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params=params, lr=base_lr, betas=(0.9, 0.999))
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params=params, lr=base_lr, momentum=0.9)
        else:
            raise RuntimeError(f"Unrecongnized optimizer type {args.optimizer}")
        return optimizer

    def get_lrate(self, step):
        args = self.args
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000

        scaling = (decay_rate ** (step / decay_steps))
        base_lrate = args.lrate * scaling
        vert_lrate = args.lrate * args.optimize_verts_gain * scaling
        name_lrates = [("lr", base_lrate), ("vertlr", vert_lrate)]
        return name_lrates

    def update_step(self, step):
        if step >= self.args.optimize_geo_start:
            self.optimize_geometry = True

        # decide upsample
        # if step in self.upsample_stage:
        #     scaling = 0.5 ** (len(self.upsample_stage) - self.upsample_stage.index(step) - 1)
        #     scaled_size = int(self.atlas_full_h * scaling), int(self.atlas_full_w * scaling)
        #     print(f"  Upsample to {scaled_size} in step {step}")
        #     self.register_parameter("atlas",
        #                             nn.Parameter(
        #                                 torchf.upsample(self.atlas, scaled_size, mode='bilinear'),
        #                                 requires_grad=True))
        #     with torch.no_grad():
        #         uv_scaling = torch.tensor([
        #             (scaled_size[1] - 1) / (self.atlas.shape[-1] - 1),
        #             (scaled_size[0] - 1) / (self.atlas.shape[-2] - 1),
        #         ]).reshape(-1, 2).type_as(self.uvs)
        #         self.uvs *= uv_scaling

    def init_from_mpi(self, state_dict):
        self._verts.data = state_dict['_verts'].type_as(self._verts)
        self.uvs.data = state_dict['uvs'].type_as(self.uvs)
        self.atlas.data = state_dict['atlas'].type_as(self.atlas)
        self.uvfaces.data = state_dict['uvfaces'].type_as(self.uvfaces)
        self.faces.data = state_dict['faces'].type_as(self.faces)
        self.ref_extrin.data = state_dict['ref_extrin'].type_as(self.ref_extrin)
        self.ref_intrin.data = state_dict['ref_intrin'].type_as(self.ref_intrin)
        self.planedepth.data = state_dict['planedepth'].type_as(self.planedepth)
        self.is_sparse = state_dict["self.is_sparse"]
        self.atlas_full_w = state_dict["self.atlas_full_w"]
        self.atlas_full_h = state_dict["self.atlas_full_h"]
        self.atlas_grid_h = state_dict["self.atlas_grid_h"]
        self.atlas_grid_w = state_dict["self.atlas_grid_w"]

        if "self.has_dyn" in state_dict.keys():
            self.has_dyn = state_dict["self.has_dyn"]
            self.atlas_full_dyn_w = state_dict["self.atlas_full_dyn_w"]
            self.atlas_full_dyn_h = state_dict["self.atlas_full_dyn_h"]
            self.atlas_grid_dyn_h = state_dict["self.atlas_grid_dyn_h"]
            self.atlas_grid_dyn_w = state_dict["self.atlas_grid_dyn_w"]
            uvs_dyn = state_dict['uvs_dyn'].type_as(self.uvs)
            uvfaces_dyn = state_dict['uvfaces_dyn'].type_as(self.uvfaces)
            faces_dyn = state_dict['faces_dyn'].type_as(self.faces)
            atlas_dyn = state_dict['atlas_dyn'].type_as(self.atlas)
            self.register_parameter("uvs_dyn", nn.Parameter(uvs_dyn, requires_grad=True))
            self.register_buffer("uvfaces_dyn", uvfaces_dyn)
            self.register_buffer("faces_dyn", faces_dyn)
            self.register_parameter("atlas_dyn", nn.Parameter(atlas_dyn, requires_grad=True))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict()
        state_dict["self.is_sparse"] = self.is_sparse
        state_dict["self.atlas_full_w"] = self.atlas_full_w
        state_dict["self.atlas_full_h"] = self.atlas_full_h
        state_dict["self.atlas_grid_h"] = self.atlas_grid_h
        state_dict["self.atlas_grid_w"] = self.atlas_grid_w

        if hasattr(self, "atlas_dyn"):
            state_dict["self.has_dyn"] = self.has_dyn
            state_dict["self.atlas_full_dyn_w"] = self.atlas_full_dyn_w
            state_dict["self.atlas_full_dyn_h"] = self.atlas_full_dyn_h
            state_dict["self.atlas_grid_dyn_h"] = self.atlas_grid_dyn_h
            state_dict["self.atlas_grid_dyn_w"] = self.atlas_grid_dyn_w
        return state_dict

    def save_mesh(self, prefix):
        vertices, faces, uvs = self.verts.detach(), self.faces.detach(), self.uvs.detach()
        mesh1 = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy())
        txt = mesh1.export(prefix + ".obj", "obj")
        with open(prefix + ".obj", 'w') as f:
            f.write(txt)

        if self.has_dyn:
            faces_dyn = self.faces_dyn
            mesh_dyn = trimesh.Trimesh(vertices.cpu().numpy(), faces_dyn.cpu().numpy())
            txt = mesh_dyn.export(prefix + "_dyn.obj", "obj")
            with open(prefix + "_dyn.obj", 'w') as f:
                f.write(txt)

    @torch.no_grad()
    def save_texture(self, prefix):
        _, atlas_cnl, atlas_h, atlas_w = self.atlas.shape
        texture = self.atlas.detach()[0].permute(1, 2, 0).reshape(-1, self.atlas.shape[1])
        ray_dir = torch.tensor([[0, 0, 1.]]).type_as(texture).expand(len(texture), -1)
        ray_dir = self.view_embed_fn(ray_dir)
        tex_input = torch.cat([texture, ray_dir], dim=-1)
        chunksz = self.args.chunk
        rgba = torch.cat([self.feat2rgba(tex_input[batchi: batchi + chunksz])
                          for batchi in range(0, len(tex_input), chunksz)])
        rgba = torch.cat([self.rgb_activate(rgba[..., :-1]), self.alpha_activate(rgba[..., -1:])], dim=-1)
        texture = (rgba * 255).type(torch.uint8).reshape(atlas_h, atlas_w, 4).cpu().numpy()
        import imageio
        imageio.imwrite(prefix + ".png", texture)

        if self.has_dyn:
            tex_dyn = self.atlas_dyn.detach()[0].permute(1, 2, 0)
            rgba = torch.cat([self.rgb_activate(tex_dyn[..., :-1]), self.alpha_activate(tex_dyn[..., -1:])], dim=-1)
            tex_dyn = (rgba * 255).type(torch.uint8).cpu().numpy()
            imageio.imwrite(prefix + "_dyn.png", tex_dyn)

    @torch.no_grad()
    def save_loopmask(self, prefix):
        if not self.args.learn_loop_mask:
            alpha = self.alpha_activate(self.atlas[0, -1:])
            loopmask = torch.sigmoid(self.atlas_mask[0])
            zero = torch.zeros_like(loopmask)
            rgba = torch.cat([1 - loopmask, loopmask, zero, alpha]).permute(1, 2, 0)
            rgba = (rgba * 255).type(torch.uint8).cpu().numpy()
            import imageio
            imageio.imwrite(prefix + ".png", rgba)

    @torch.no_grad()
    def direct2sh(self):
        self.view_embed_fn, self.view_cnl = get_embedder(self.args.multires_views, input_dim=3)
        self.rgb_mlp_type = 'rgb_sh'
        atlas_cnl = 3 * 4 + 1  # 9 for each channel
        atlas = torch.zeros((1, atlas_cnl, self.atlas.shape[-2], self.atlas.shape[-1])).type_as(self.atlas)
        atlas[:, -1] = self.atlas.data[:, -1]
        atlas[:, 0:-1:4] = self.atlas.data[:, :3]
        self.register_parameter("atlas", nn.Parameter(atlas, requires_grad=True))

        self.feat2rgba = SphericalHarmoic_RGB(atlas_cnl, self.view_cnl)
        self.use_viewdirs = True

    @torch.no_grad()
    def sparsify_faces(self, erode_num=2, alpha_thresh=0.03, loop_thresh=0.5):
        print("Sparsifying the faces")
        # faces of a quad: 0 - 1
        #                  | \ |
        #                  2 - 3
        quads = self.uvfaces.reshape(-1, 6)  # (6) is two faces of (0, 1, 3), (3, 2, 0)
        assert (quads[:, 0] == quads[:, 5]).all() and (quads[:, 2] == quads[:, 3]).all()
        uvs = self.uvs
        # decide size of a quad in atlas space
        atlas_h, atlas_w = self.atlas.shape[-2:]
        uvsz_w = (uvs[quads[0, 1]] - uvs[quads[0, 0]])[0].item()
        uvsz_h = (uvs[quads[0, 4]] - uvs[quads[0, 0]])[1].item()
        imsz_w = int(np.round(uvsz_w / 2 * (atlas_w - 1)))
        imsz_h = int(np.round(uvsz_h / 2 * (atlas_h - 1)))
        grid_offset = torch.meshgrid([torch.linspace(0, uvsz_h, imsz_h), torch.linspace(0, uvsz_w, imsz_w)])
        grid_offset = torch.stack(grid_offset[::-1], dim=-1)[None].type_as(self.atlas)  # 1, h, w, 2

        quad_v0 = quads[:, 0]
        n_quad = len(quad_v0)
        uv_v0 = uvs[quad_v0][:, None, None, :]  # B, 1, 1, 2
        grid = uv_v0 + grid_offset

        # get alpha atlas for deciding mask
        texture = self.atlas.detach()[0].permute(1, 2, 0).reshape(-1, self.atlas.shape[1])
        ray_dir = torch.tensor([[0, 0, 1.]]).type_as(texture).expand(len(texture), -1)
        ray_dir = self.view_embed_fn(ray_dir)
        tex_input = torch.cat([texture, ray_dir], dim=-1)
        chunksz = self.args.chunk
        alpha = torch.cat([self.feat2rgba(tex_input[batchi: batchi + chunksz])[..., -1:]
                           for batchi in range(0, len(tex_input), chunksz)])
        alpha[alpha == ALPHA_INIT_VAL] = -10
        alpha = self.alpha_activate(alpha).reshape(1, 1, *self.atlas.shape[-2:])  # (1, 1, H, W)
        loopmask = self.atlas_mask
        loopmask[loopmask == ALPHA_INIT_VAL] = -10
        loopmask = torch.sigmoid(loopmask).reshape(1, 1, *self.atlas.shape[-2:])

        for i in range(erode_num):
            loopmask = erode(loopmask)
        for i in range(erode_num):
            loopmask = dilate(loopmask)

        for i in range(erode_num):
            alpha = erode(alpha)
        for i in range(erode_num + 1):
            alpha = dilate(alpha)

        # sample to batches
        # !!!!!!!!!!!Align_corners=True indicates that -1 is the left-most pixel, 1 is the right-most pixel
        atlases = torchf.grid_sample(self.atlas.expand(n_quad, -1, -1, -1), grid, align_corners=True)
        atlases = atlases.permute(0, 2, 3, 1)

        atlases_alpha = torchf.grid_sample(alpha.expand(n_quad, -1, -1, -1), grid, align_corners=True)
        atlases_alpha = atlases_alpha.permute(0, 2, 3, 1)
        if self.args.sparsify_rmfirstlayer > 0:
            print("INFO::remove the first layer when sparsify")
            n_quad_perlayer = self.mpi_h_verts * self.mpi_w_verts * self.args.sparsify_rmfirstlayer
            atlases_alpha[:n_quad_perlayer] = 0
        atlases_alpha = atlases_alpha.reshape(n_quad, -1)
        atlases_loop = torchf.grid_sample(loopmask.expand(n_quad, -1, -1, -1), grid, align_corners=True)
        atlases_loop = atlases_loop.permute(0, 2, 3, 1)
        atlases_loop = atlases_loop.reshape(n_quad, -1)

        mask = (atlases_alpha.max(dim=-1)[0] > alpha_thresh)
        mask_loop = (atlases_loop.max(dim=-1)[0] > loop_thresh)

        # mask[::10] = True
        # mask_loop[::20] = True
        # # TODO: delete this

        mask_loop = torch.logical_and(mask, mask_loop)
        mask_loop_short = mask_loop[mask]
        n_mask = torch.count_nonzero(mask).item()
        n_dyn = torch.count_nonzero(mask_loop).item()
        n_static = n_mask - n_dyn

        # decide atlas hight and width
        max_ratio = 4  # the max value of width / height

        def get_hw(n):
            n_min = int(np.sqrt(n / max_ratio))
            n_max = int(np.sqrt(n))
            n_try = np.arange(n_min, n_max)
            selected = np.argmin(n_try - n % n_try)
            _h = n_try[selected]
            _w = n // _h + 1
            _res = _h * _w - n
            return _h, _w, _res

        n_height_static, n_width_static, n_residual_static = get_hw(n_static)
        n_height_dyn, n_width_dyn, n_residual_dyn = get_hw(n_dyn)

        print(f"mask {n_mask} / {n_quad} ({100 * n_mask / n_quad:.2f}%) quads")
        print(f"   of {n_mask}, {n_dyn} ({100 * n_dyn / n_mask:.2f}%) is dynamic quads")

        # update atlas
        new_atlas = atlases.reshape(n_quad, imsz_h, imsz_w, -1)[mask, ...]
        new_faces = self.faces.reshape(-1, 2, 3)[mask, ...]

        new_atlas_static = new_atlas[torch.logical_not(mask_loop_short), ...]
        new_faces_static = new_faces[torch.logical_not(mask_loop_short), ...].reshape(-1, 3)
        new_atlas_static = torch.cat([new_atlas_static, new_atlas_static[-1:].expand(n_residual_static, -1, -1, -1)])
        new_atlas_static = new_atlas_static.reshape(n_height_static, n_width_static, imsz_h, imsz_w, -1)\
            .permute(0, 2, 1, 3, 4).reshape(1, n_height_static * imsz_h, n_width_static * imsz_w, -1)\
            .permute(0, 3, 1, 2)

        new_atlas_dyn = new_atlas[mask_loop_short, ...]
        new_faces_dyn = new_faces[mask_loop_short, ...].reshape(-1, 3)
        new_atlas_dyn = torch.cat([new_atlas_dyn, new_atlas_dyn[-1:].expand(n_residual_dyn, -1, -1, -1)])
        new_atlas_dyn = new_atlas_dyn.reshape(n_height_dyn, n_width_dyn, imsz_h, imsz_w, -1) \
            .permute(0, 2, 1, 3, 4).reshape(1, n_height_dyn * imsz_h, n_width_dyn * imsz_w, -1)\
            .permute(0, 3, 1, 2)

        def gen_quad_uvs(atlash, atlasw, ntile):
            # update faces, uvfaces, uvs
            quad_uvsz_h, quad_uvsz_w = 2 / (atlash - 1) * (imsz_h - 1), 2 / (atlasw - 1) * (imsz_w - 1)
            uvs_offset = torch.tensor([[0, 0], [quad_uvsz_w, 0],
                                       [0, quad_uvsz_h], [quad_uvsz_w, quad_uvsz_h]]).type_as(self.uvs)
            quad_uv0 = torch.meshgrid(
                torch.arange(0, atlash, imsz_h) / (atlash - 1) * 2 - 1,
                torch.arange(0, atlasw, imsz_h) / (atlasw - 1) * 2 - 1
            )
            quad_uv0 = torch.stack(quad_uv0[::-1], dim=-1).type_as(self.uvs)
            quad_uvs = quad_uv0[:, :, None, :] + uvs_offset[None, None, :, :]
            quad_uvs = quad_uvs.reshape(-1, 4, 2)[:ntile]
            uvid_offset = torch.tensor([[0, 1, 3], [3, 2, 0]]).type_as(self.uvfaces)
            uvid0 = torch.arange(ntile).type_as(self.uvfaces) * 4
            uvfaces = uvid0[:, None, None] + uvid_offset[None]
            return quad_uvs, uvfaces

        atlas_h, atlas_w = new_atlas_static.shape[-2:]
        quad_uvs_static, uvfaces_static = gen_quad_uvs(atlas_h, atlas_w, n_static)
        atlas_h, atlas_w = new_atlas_dyn.shape[-2:]
        quad_uvs_dyn, uvfaces_dyn = gen_quad_uvs(atlas_h, atlas_w, n_dyn)

        self.is_sparse = True
        self.atlas_grid_h, self.atlas_grid_w = n_height_static, n_width_static
        self.atlas_full_h, self.atlas_full_w = new_atlas_static.shape[-2:]
        self.atlas_grid_dyn_h, self.atlas_grid_dyn_w = n_height_dyn, n_width_dyn
        self.atlas_full_dyn_h, self.atlas_full_dyn_w = new_atlas_dyn.shape[-2:]
        self.register_parameter("uvs", nn.Parameter(quad_uvs_static.reshape(-1, 2), requires_grad=True))
        self.register_buffer("uvfaces", uvfaces_static.reshape(-1, 3).long())
        self.register_buffer("faces", new_faces_static.long())
        self.register_parameter("atlas", nn.Parameter(new_atlas_static, requires_grad=True))

        self.has_dyn = True
        self.register_parameter("uvs_dyn", nn.Parameter(quad_uvs_dyn.reshape(-1, 2), requires_grad=True))
        self.register_buffer("uvfaces_dyn", uvfaces_dyn.reshape(-1, 3).long())
        self.register_buffer("faces_dyn", new_faces_dyn.long())
        self.register_parameter("atlas_dyn", nn.Parameter(new_atlas_dyn, requires_grad=True))

        self.args.learn_loop_mask = False
        del self.atlas_mask

    @property
    def verts(self):
        verts = self._verts
        if self.args.normalize_verts:
            depth_scaling = self.planedepth
            verts = (verts.reshape(len(depth_scaling), -1) * depth_scaling[:, None]).reshape_as(verts)
        return verts

    def render(self, H, W, extrin, intrin):
        B = len(extrin)
        verts = self.verts.reshape(1, -1, 3)
        with torch.set_grad_enabled(self.optimize_geometry):
            R, T = extrin[:, :3, :3], extrin[:, :3, 3]
            # normalize intrin to ndc
            intrin_ptc = intrin.clone()
            if H < W:  # strange trick to make raster result correct
                intrin_ptc[:, :2] *= (- 2 / H)
                intrin_ptc[:, 0, 2] += W / H
                intrin_ptc[:, 1, 2] += 1
            else:
                intrin_ptc[:, :2] *= (- 2 / W)
                intrin_ptc[:, 0, 2] += 1
                intrin_ptc[:, 1, 2] += H / W

            # transform to ndc space
            vert_view = (R @ verts[..., None] + T[..., None])
            vert_ndc = (intrin_ptc[:, :3, :3] @ vert_view)[..., 0]
            vert_ndc = vert_ndc[..., :2] / vert_ndc[..., 2:]
            vert = torch.cat([vert_ndc[..., :2], vert_view[..., 2:3, 0]], dim=-1)

            # rasterize
            raster_settings = RasterizationSettings(
                image_size=(H, W),  # viewport
                blur_radius=0.0,
                faces_per_pixel=self.mpi_d,
            )
            raster = SimpleRasterizer(raster_settings)
            static_face_count = len(self.faces)
            if self.has_dyn:
                faces = torch.cat([self.faces, self.faces_dyn]).reshape(1, -1, 3)
            else:
                faces = self.faces.reshape(1, -1, 3)
            frag: Fragments = raster(
                vert, faces
            )
            pixel_to_face, depths, bary_coords = frag.pix_to_face, frag.zbuf, frag.bary_coords
            depths = torch.reciprocal(depths)
            depths = (depths - 1 / self.far) / (1 / self.near - 1 / self.far)
            num_layers = pixel_to_face.shape[-1]
            # currently the batching is not supported
            mask = torch.logical_and(pixel_to_face >= 0, pixel_to_face < static_face_count)
            mask_dyn = pixel_to_face >= static_face_count
            mask_flat = mask.reshape(-1)
            mask_dyn_flat = mask_dyn.reshape(-1)

            def get_uvs(mask_flat_, uvs_, uvfaces_, offset_=0):
                faces_ma_ = pixel_to_face.reshape(-1)[mask_flat_] - offset_
                uv_indices = uvfaces_[faces_ma_]
                uvs = uvs_[uv_indices]  # N, 3, n_feat
                bary_coords_ma = bary_coords.reshape(-1, 3)[mask_flat_, :]  # N, 3
                uvs = (bary_coords_ma[..., None] * uvs).sum(dim=-2)
                return uvs

            uvs = get_uvs(mask_flat, self.uvs, self.uvfaces)

        _, ray_direction = get_rays_tensor_batches(H, W, intrin, pose2extrin_torch(extrin))
        ray_direction = ray_direction / ray_direction.norm(dim=-1, keepdim=True)

        def render_masked_rgba(mask_, atlas_, uvs_):
            mask_flat_ = mask_.reshape(-1)
            ray_direction_ = ray_direction[..., None, :].expand(mask_.shape + (3,))
            ray_d_ = ray_direction_.reshape(-1, 3)[mask_flat_, :]
            ray_d_ = self.view_embed_fn(ray_d_.reshape(-1, 3))
            rgba_feat_ = torchf.grid_sample(atlas_,
                                            uvs_[None, None, ...],
                                            padding_mode="zeros", align_corners=True)
            rgba_feat_ = rgba_feat_.reshape(atlas_.shape[1], -1).permute(1, 0)

            chunksz = self.args.chunk
            tex_input_ = torch.cat([rgba_feat_, ray_d_], dim=-1)
            rgba_ = self.feat2rgba(tex_input_)
            rgba_ = torch.cat([self.rgb_activate(rgba_[..., :-1]), self.alpha_activate(rgba_[..., -1:])], dim=-1)
            return rgba_

        rgba = render_masked_rgba(mask, self.atlas, uvs)

        canvas = torch.zeros((B, H, W, num_layers, 4)).type_as(rgba).reshape(-1, 4)
        mpi = torch.masked_scatter(canvas, mask_flat[:, None].expand_as(canvas), rgba)
        if self.has_dyn:
            with torch.no_grad():
                uvs_dyn = get_uvs(mask_dyn_flat, self.uvs_dyn, self.uvfaces_dyn, offset_=static_face_count)
            rgba_dyn = render_masked_rgba(mask_dyn, self.atlas_dyn, uvs_dyn)
            mpi = torch.masked_scatter(mpi, mask_dyn_flat[:, None].expand_as(canvas), rgba_dyn)
        mpi = mpi.reshape(B, H, W, num_layers, 4)
        # make rgb d a plane
        rgb, blend_weight = overcompose(
            mpi[..., -1], mpi[..., :-1]
        )
        alpha = blend_weight.sum(dim=-1)
        if len(self.args.bg_color) > 0:
            r, g, b = map(float, self.args.bg_color.split('#'))
            bg_color = torch.tensor([r, g, b]).type_as(rgb)
            rgb = rgb * alpha[..., None] + bg_color[None, None, None] * (- alpha[..., None] + 1)

        # get depth map
        if self.args.normalize_blendweight_fordepth:
            blend_weight = blend_weight / alpha.clamp_min(1e-10)[..., None]
        disp = (depths * blend_weight).sum(-1)

        if self.args.learn_loop_mask:
            assert not self.has_dyn
            label = torchf.grid_sample(self.atlas_mask,
                                       uvs[None, None, ...],
                                       padding_mode="zeros", align_corners=True)
            label = torch.sigmoid(label)
            canvas = torch.zeros((B, H, W, num_layers, 1)).type_as(rgba).reshape(-1, 1)
            mpi_mask = torch.masked_scatter(canvas, mask_flat[:, None].expand_as(canvas), label)
            mpi_mask = mpi_mask.reshape(B, H, W, num_layers, 1)
            label, _ = overcompose(
                mpi[..., -1].detach(), mpi_mask  # detach mpi so that geometry is not related to mpi
            )
            rgbl = torch.cat([rgb, label], dim=-1)
        else:
            rgbl = rgb

        variables = {
            "pix_to_face": pixel_to_face,
            "blend_weight": blend_weight,
            "mpi": mpi,
            "disp_norm": disp,
            "alpha": alpha
        }

        return rgbl, variables

    def forward(self, h, w, tar_extrins, tar_intrins):
        extrins = tar_extrins @ self.ref_extrin[None, ...].inverse()

        rgbl, variables = self.render(h, w, extrins, tar_intrins)
        rgbl = rgbl.permute(0, 3, 1, 2)
        extra = {}
        if self.training:
            if self.args.sparsity_loss_weight > 0:
                alpha = variables["mpi"][..., -1]
                sparsity = alpha.norm(dim=-1, p=1) / alpha.norm(dim=-1, p=2).clamp_min(1e-6)
                sparsity = sparsity.mean() / np.sqrt(self.mpi_d)  # so it's inrrelvant to the layer num
                extra["sparsity"] = sparsity.reshape(1, -1)

            if self.args.rgb_smooth_loss_weight > 0:
                smooth = variables["mpi"][..., :-1]
                denorm = smooth.shape[-2] / self.mpi_d
                smoothx = (smooth[:, :, :-1] - smooth[:, :, 1:]).abs().mean()
                smoothy = (smooth[:, :-1] - smooth[:, 1:]).abs().mean()
                smooth = (smoothx + smoothy).reshape(1, -1) * denorm
                extra["rgb_smooth"] = smooth.reshape(1, -1)

            if self.args.a_smooth_loss_weight > 0:
                smooth = variables["mpi"][..., -1]
                denorm = smooth.shape[-1] / self.mpi_d
                smoothx = (smooth[:, :, :-1] - smooth[:, :, 1:]).abs().mean()
                smoothy = (smooth[:, :-1] - smooth[:, 1:]).abs().mean()
                smooth = (smoothx + smoothy)
                extra["a_smooth"] = smooth.reshape(1, -1) * denorm

            if self.args.d_smooth_loss_weight > 0:
                disp = variables['disp_norm']
                depth_gradx = (disp[:, 1:, :-1] - disp[:, 1:, 1:]).abs()
                depth_grady = (disp[:, :-1, 1:] - disp[:, 1:, 1:]).abs()
                depth_grad = depth_gradx + depth_grady

                rgb = rgbl[:, :3]
                rgb_gradx = (rgb[..., 1:, :-1] - rgb[..., 1:, 1:]).abs().sum(dim=1)
                rgb_grady = (rgb[..., :-1, 1:] - rgb[..., 1:, 1:]).abs().sum(dim=1)
                edge = rgb_gradx + rgb_grady
                weight = (- edge * self.args.edge_scale + 1).clamp_min(0)
                d_smooth = (depth_grad * weight).mean()
                extra["d_smooth"] = d_smooth.reshape(1, -1)

            if self.args.density_loss_weight > 0:
                alpha = variables["alpha"]
                density = (alpha - 1).abs().mean()
                extra["density"] = density.reshape(1, -1)

            if self.args.laplacian_loss_weight > 0:
                verts = self.verts.reshape(self.mpi_d, self.mpi_h_verts, self.mpi_w_verts, -1)
                verts_pad = torch.cat([
                    verts[:, :, :1] * 2 - verts[:, :, 1:2], verts, verts[:, :, -1:] * 2 - verts[:, :, -2:-1]
                ], dim=2)
                verts_pad = torch.cat([
                    verts_pad[:, :1] * 2 - verts_pad[:, 1:2], verts_pad, verts_pad[:, -1:] * 2 - verts_pad[:, -2:-1]
                ], dim=1)
                verts_laplacian_x = (verts_pad[:, :-2, 1:-1] + verts_pad[:, 2:, 1:-1]) / 2
                verts_laplacian_y = (verts_pad[:, 1:-1, :-2] + verts_pad[:, 1:-1, 2:]) / 2
                verts_laplacian = (verts_laplacian_y - verts).norm(dim=-1) + (verts_laplacian_x - verts).norm(dim=-1)
                extra["laplacian"] = verts_laplacian.mean().reshape(1, -1)
        return rgbl, extra
