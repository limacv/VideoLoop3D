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
        elif args.rgb_mlp_type == "rgba_sh":
            atlas_cnl = 4 * 4  # 9 for each channel
            atlas = torch.rand((1, atlas_cnl, int(self.atlas_full_h), int(self.atlas_full_w)))
            self.feat2rgba = SphericalHarmoic_RGBA(atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
        else:
            raise RuntimeError(f"rgbmlp_type = {args.rgb_mlp_type} not recognized")

        self.register_parameter("atlas", nn.Parameter(atlas, requires_grad=True))
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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict()
        state_dict["self.is_sparse"] = self.is_sparse
        state_dict["self.atlas_full_w"] = self.atlas_full_w
        state_dict["self.atlas_full_h"] = self.atlas_full_h
        state_dict["self.atlas_grid_h"] = self.atlas_grid_h
        state_dict["self.atlas_grid_w"] = self.atlas_grid_w
        return state_dict

    def save_mesh(self, prefix):
        vertices, faces, uvs = self.verts.detach(), self.faces.detach(), self.uvs.detach()
        # uv_scaling = torch.tensor([
        #     1 / (self.atlas.shape[-1] - 1),
        #     1 / (self.atlas.shape[-2] - 1)
        # ])
        # color = torch.cat([1 - uvs * uv_scaling, torch.zeros_like(uvs[:, :1])], dim=-1)
        color = torch.cat([0.5 - uvs * 0.5, torch.zeros_like(uvs[:, :1])], dim=-1)
        color = np.clip(color.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mesh1 = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(),
                                vertex_colors=color)
        txt = mesh1.export(prefix + ".obj", "obj")
        with open(prefix + ".obj", 'w') as f:
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

    @torch.no_grad()
    def direct2sh(self):
        self.view_embed_fn, self.view_cnl = get_embedder(self.args.multires_views, input_dim=3)
        self.rgb_mlp_type = 'rgb_sh'
        atlas_cnl = 3 * 4 + 1  # 9 for each channel
        atlas = torch.zeros((1, atlas_cnl, self.atlas.shape[-2], self.atlas.shape[-1])).type_as(self.atlas)
        atlas[:, 0] = self.atlas.data[:, -1]
        atlas[:, 1::4] = self.atlas.data[:, :3]
        self.register_parameter("atlas", nn.Parameter(atlas, requires_grad=True))

        self.feat2rgba = SphericalHarmoic_RGB(atlas_cnl, self.view_cnl)
        self.use_viewdirs = True

    @torch.no_grad()
    def sparsify_faces(self, alpha_thresh=0.03):
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
        for i in range(3):
            alpha = erode(alpha)
        for i in range(5):
            alpha = dilate(alpha)

        # sample to batches
        # !!!!!!!!!!!Align_corners=True indicates that -1 is the left-most pixel, 1 is the right-most pixel
        atlases = torchf.grid_sample(self.atlas.expand(n_quad, -1, -1, -1), grid, align_corners=True)
        atlases = atlases.permute(0, 2, 3, 1)

        atlases_alpha = torchf.grid_sample(alpha.expand(n_quad, -1, -1, -1), grid, align_corners=True)
        atlases_alpha = atlases_alpha.permute(0, 2, 3, 1)
        atlases_alpha = atlases_alpha.reshape(n_quad, -1)
        mask = (atlases_alpha.max(dim=-1)[0] > alpha_thresh)
        n_mask = torch.count_nonzero(mask).item()

        # decide atlas hight and width
        max_ratio = 4  # the max value of width / height
        n_min = int(np.sqrt(n_mask / max_ratio))
        n_max = int(np.sqrt(n_mask))
        n_try = np.arange(n_min, n_max)
        selected = np.argmin(n_try - n_mask % n_try)
        n_height = n_try[selected]
        n_width = n_mask // n_height + 1
        n_residual = n_height * n_width - n_mask

        print(f"mask {n_mask} / {n_quad} ({100 * n_mask / n_quad:.2f}%) quads")
        # update atlas
        new_atlas = atlases.reshape(n_quad, imsz_h, imsz_w, -1)[mask, ...]
        new_atlas_pad = torch.cat([new_atlas, new_atlas[-1:].expand(n_residual, -1, -1, -1)])
        new_atlas = new_atlas_pad.reshape(n_height, n_width, imsz_h, imsz_w, -1).permute(0, 2, 1, 3, 4)
        new_atlas = new_atlas.reshape(1, n_height * imsz_h, n_width * imsz_w, -1).permute(0, 3, 1, 2)
        atlas_h, atlas_w = new_atlas.shape[-2:]

        # update faces, uvfaces, uvs
        new_faces = self.faces.reshape(-1, 2, 3)[mask, ...].reshape(-1, 3)
        quad_uvsz_h, quad_uvsz_w = 2 / (atlas_h - 1) * (imsz_h - 1), 2 / (atlas_w - 1) * (imsz_w - 1)
        uvs_offset = torch.tensor([[0, 0], [quad_uvsz_w, 0],
                                   [0, quad_uvsz_h], [quad_uvsz_w, quad_uvsz_h]]).type_as(self.uvs)
        quad_uv0 = torch.meshgrid(
            torch.arange(0, atlas_h, imsz_h) / (atlas_h - 1) * 2 - 1,
            torch.arange(0, atlas_w, imsz_h) / (atlas_w - 1) * 2 - 1
        )
        quad_uv0 = torch.stack(quad_uv0[::-1], dim=-1).type_as(self.uvs)
        quad_uvs = quad_uv0[:, :, None, :] + uvs_offset[None, None, :, :]
        quad_uvs = quad_uvs.reshape(-1, 4, 2)[:-n_residual]
        uvid_offset = torch.tensor([[0, 1, 3], [3, 2, 0]]).type_as(self.uvfaces)
        uvid0 = torch.arange(n_mask).type_as(self.uvfaces) * 4
        uvfaces = uvid0[:, None, None] + uvid_offset[None]

        self.is_sparse = True
        self.atlas_grid_h, self.atlas_grid_w = n_height, n_width
        self.atlas_full_h, self.atlas_full_w = n_height * imsz_h, n_width * imsz_w
        self.register_parameter("uvs", nn.Parameter(quad_uvs.reshape(-1, 2), requires_grad=True))
        self.register_buffer("uvfaces", uvfaces.reshape(-1, 3).long())
        self.register_buffer("faces", new_faces.long())
        self.register_parameter("atlas", nn.Parameter(new_atlas, requires_grad=True))

    @property
    def verts(self):
        verts = self._verts
        if self.args.normalize_verts:
            depth_scaling = self.planedepth
            verts = (verts.reshape(len(depth_scaling), -1) * depth_scaling[:, None]).reshape_as(verts)
        return verts

    def render(self, H, W, extrin, intrin):
        B = len(extrin)
        verts, faces = self.verts.reshape(1, -1, 3), self.faces.reshape(1, -1, 3)
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
            frag: Fragments = raster(
                vert, faces
            )
            pixel_to_face, depths, bary_coords = frag.pix_to_face, frag.zbuf, frag.bary_coords

            # currently the batching is not supported
            mask = pixel_to_face.reshape(-1) >= 0
            faces_ma = pixel_to_face.reshape(-1)[mask]
            uv_indices = self.uvfaces[faces_ma]
            uvs = self.uvs[uv_indices]  # N, 3, n_feat
            bary_coords_ma = bary_coords.reshape(-1, 3)[mask, :]  # N, 3
            uvs = (bary_coords_ma[..., None] * uvs).sum(dim=-2)

        _, ray_direction = get_rays_tensor_batches(H, W, intrin, pose2extrin_torch(extrin))
        ray_direction = ray_direction / ray_direction.norm(dim=-1, keepdim=True)
        ray_direction = ray_direction[..., None, :].expand(pixel_to_face.shape + (3,))
        ray_d = ray_direction.reshape(-1, 3)[mask, :]
        ray_d = self.view_embed_fn(ray_d.reshape(-1, 3))

        # uv from 0, S - 1  to -1, 1
        # uv_scaling = torch.tensor([
        #     2 / (self.atlas.shape[-1] - 1),
        #     2 / (self.atlas.shape[-2] - 1)
        # ])
        # uvs = uvs * uv_scaling - 1

        rgba_feat = torchf.grid_sample(self.atlas,
                                       uvs[None, None, ...],
                                       padding_mode="zeros", align_corners=True)
        rgba_feat = rgba_feat.reshape(self.atlas.shape[1], -1).permute(1, 0)
        chunksz = self.args.chunk
        tex_input = torch.cat([rgba_feat, ray_d], dim=-1)
        rgba = torch.cat([self.feat2rgba(tex_input[batchi: batchi + chunksz])
                          for batchi in range(0, len(tex_input), chunksz)])
        rgba = torch.cat([self.rgb_activate(rgba[..., :-1]), self.alpha_activate(rgba[..., -1:])], dim=-1)
        canvas = torch.zeros((B, H, W, self.mpi_d, 4)).type_as(rgba).reshape(-1, 4)
        mpi = torch.masked_scatter(canvas, mask[:, None].expand_as(canvas), rgba)
        mpi = mpi.reshape(B, H, W, self.mpi_d, 4)
        # make rgb d a plane
        rgbd, blend_weight = overcompose(
            mpi[..., -1],
            torch.cat([mpi[..., :-1], depths[..., None]], dim=-1)
        )

        variables = {
            "pix_to_face": pixel_to_face,
            "blend_weight": blend_weight,
            "mpi": mpi,
            "depth": rgbd[..., -1],
            "alpha": blend_weight.sum(dim=-1)
        }

        return rgbd[..., :3], variables

    def forward(self, h, w, tar_extrins, tar_intrins):
        extrins = tar_extrins @ self.ref_extrin[None, ...].inverse()

        rgb, variables = self.render(h, w, extrins, tar_intrins)
        rgb = rgb.permute(0, 3, 1, 2)
        extra = {}
        if self.training:
            if self.args.sparsity_loss_weight > 0:
                alpha = variables["mpi"][..., -1]
                sparsity = alpha.norm(dim=-1, p=1) / alpha.norm(dim=-1, p=2).clamp_min(1e-6)
                sparsity = sparsity.mean() / np.sqrt(alpha.shape[-1])  # so it's inrrelvant to the layer num
                extra["sparsity"] = sparsity.reshape(1, -1)

            if self.args.rgb_smooth_loss_weight > 0:
                smooth = variables["mpi"][..., :-1]
                smoothx = (smooth[:, :, :-1] - smooth[:, :, 1:]).abs().mean()
                smoothy = (smooth[:, :-1] - smooth[:, 1:]).abs().mean()
                smooth = (smoothx + smoothy).reshape(1, -1)
                extra["rgb_smooth"] = smooth.reshape(1, -1)

            if self.args.a_smooth_loss_weight > 0:
                smooth = variables["mpi"][..., -1]
                smoothx = (smooth[:, :, :-1] - smooth[:, :, 1:]).abs().mean()
                smoothy = (smooth[:, :-1] - smooth[:, 1:]).abs().mean()
                smooth = (smoothx + smoothy)
                extra["a_smooth"] = smooth.reshape(1, -1)

            if self.args.d_smooth_loss_weight > 0:
                depth = variables['depth']
                smoothx = (depth[:, :, :-1] - depth[:, :, 1:]).abs().mean()
                smoothy = (depth[:, :-1] - depth[:, 1:]).abs().mean()
                smooth = (smoothx + smoothy).reshape(1, -1)
                extra["d_smooth"] = smooth.reshape(1, -1)

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
        return rgb, extra
