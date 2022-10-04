import torch
import torch.nn as nn
import torch.nn.functional as torchf
import torchvision.transforms
import os
import imageio
import time
import cv2
from utils import *
from NeRF_modules import get_embedder
from utils_mpi import *
import trimesh
from utils_vid import Patch3DGPNNDirectLoss, Patch3DMSE, Patch3DAvg, Patch3DGPNNLowMemLoss
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
from MPI import ACTIVATES, MPMesh


class MPMeshVid(nn.Module):
    def __init__(self, args, H, W, ref_extrin, ref_intrin, near, far):
        super(MPMeshVid, self).__init__()
        self.args = args
        self.frm_num = args.mpv_frm_num
        self.isloop = args.mpv_isloop
        mpi_h, mpi_w = int(args.mpi_h_scale * H), int(args.mpi_w_scale * W)
        self.mpi_d = args.mpi_d
        self.mpi_h_verts, self.mpi_w_verts = args.mpi_h_verts, args.mpi_w_verts
        self.H, self.W = H, W
        self.atlas_grid_h, self.atlas_grid_w = args.atlas_grid_h, self.mpi_d // args.atlas_grid_h
        assert self.mpi_d % self.atlas_grid_h == 0, "mpi_d and atlas_grid_h should match"
        self.is_sparse = False  # initialize to MPI
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

        scaling = 1
        atlas = torch.rand((1, args.atlas_cnl, int(self.atlas_full_h * scaling), int(self.atlas_full_w * scaling)))
        atlas_dyn = torch.randn((self.frm_num, 4, int(self.atlas_full_h * scaling), int(self.atlas_full_w * scaling))) \
                    * args.init_std
        if args.fp16:
            atlas = atlas.half()
            atlas_dyn = atlas_dyn.half()

        # -1, 1 to 0, h
        # uvs = uvs * 0.5 + 0.5
        # atlas_size = torch.tensor([int(self.atlas_full_w * scaling), int(self.atlas_full_h * scaling)]).reshape(-1, 2)
        # uvs *= (atlas_size - 1).type_as(uvs)

        self.register_parameter("uvs", nn.Parameter(uvs, requires_grad=True))
        self.register_buffer("uvfaces", faces.clone().long())
        self._verts = nn.Parameter(verts, requires_grad=True)
        self.register_buffer("faces", faces.long())
        self.optimize_geometry = False
        self.register_parameter("atlas_dyn", nn.Parameter(atlas_dyn, requires_grad=True))
        self.register_parameter("atlas", nn.Parameter(atlas, requires_grad=True))

        self.view_embed_fn, self.view_cnl = get_embedder(args.multires_views, input_dim=3)
        self.rgb_mlp_type = args.rgb_mlp_type
        if self.rgb_mlp_type == "direct":
            self.feat2rgba = lambda x: x[..., :4]
            self.atlas.data[:, -1] = -2
            self.use_viewdirs = False
        elif self.rgb_mlp_type == "rgbamlp":
            self.feat2rgba = nn.Sequential(
                nn.Linear(self.view_cnl + args.atlas_cnl, 48), nn.ReLU(),
                nn.Linear(48, 4)
            )
            self.feat2rgba[-2].bias.data[-1] = -2
            self.use_viewdirs = True
        elif self.rgb_mlp_type == "rgbmlp":
            self.feat2rgba = Feat2RGBMLP_alpha(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
            self.atlas.data[:, 0] = -2
        elif self.rgb_mlp_type == "rgbanex":
            self.feat2rgba = NeX_RGBA(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
        elif self.rgb_mlp_type == "rgbnex":
            self.feat2rgba = NeX_RGB(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
            self.atlas.data[:, 0] = -2
        elif self.rgb_mlp_type == "rgb_sh":
            assert self.args.atlas_cnl == 3 * 9 + 1  # one for alpha, 9 for base
            self.feat2rgba = SphericalHarmoic_RGB(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
        elif self.rgb_mlp_type == "rgba_sh":
            assert self.args.atlas_cnl == 4 * 9  # 9 for each channel
            self.feat2rgba = SphericalHarmoic_RGBA(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
        else:
            raise RuntimeError(f"rgbmlp_type = {args.rgb_mlp_type} not recognized")
        self.rgb_activate = ACTIVATES[args.rgb_activate]
        self.alpha_activate = ACTIVATES[args.alpha_activate]

        # the SWD Loss
        self.swd_patch_size = args.swd_patch_size
        self.swd_patcht_size = args.swd_patcht_size
        self.swd_stride = args.swd_stride
        self.swd_stridet = args.swd_stridet

        self.losses = {
            'swd': None,
            'gpnn': Patch3DGPNNDirectLoss(),
            'gpnn_lm': Patch3DGPNNLowMemLoss(),
            'mse': Patch3DMSE,
            'avg': Patch3DAvg,
        }

    def lod(self, factor):
        if not self.is_sparse:
            h, w = int(self.atlas_full_h * factor), int(self.atlas_full_w * factor)
            print(f"MPV.lod:: Resizing the atlas from {self.atlas.shape[-2:]} to {(h, w)}")
            new_atlas = torchvision.transforms.Resize((h, w))(self.atlas.data)
            new_atlas_dyn = torchvision.transforms.Resize((h, w))(self.atlas_dyn.data)
        else:
            atlas_h, atlas_w = self.atlas.shape[-2:]
            gridh, gridw = self.atlas_grid_h, self.atlas_grid_w
            tileh, tilew = atlas_h // self.atlas_grid_h, atlas_w // self.atlas_grid_w
            fulltileh, fulltilew = self.atlas_full_h // self.atlas_grid_h, self.atlas_full_w // self.atlas_grid_w
            newtileh, newtilew = max(int(fulltileh * factor), 2), max(int(fulltilew * factor), 2)
            if newtileh == tileh and newtilew == tilew:
                print(f"MPV.lod:: no need to resize")
                return
            print(f"MPV.lod:: Sparse! Resizing the tiles from {(tileh, tilew)} to {(newtileh, newtilew)}")

            def resize_atlas(a_):
                b, c = a_.shape[:2]
                a_ = a_.reshape(b, c, gridh, tileh, gridw, tilew)
                a_ = a_.permute(0, 2, 4, 1, 3, 5)  # b, gh, gw, c, th, tw
                a_ = torchvision.transforms.Resize((newtileh, newtilew))(a_.reshape(-1, c, tileh, tilew))
                a_ = a_.reshape(b, gridh, gridw, c, newtileh, newtilew).permute(0, 3, 1, 4, 2, 5)
                return a_.reshape(b, c, gridh * newtileh, gridw * newtilew)

            new_atlas = resize_atlas(self.atlas.data)
            new_atlas_dyn = resize_atlas(self.atlas_dyn.data)

            # need to recompute the uv to prevent the anti-aliasing effect
            new_atlas_h, new_atlas_w = new_atlas.shape[-2:]
            uvs = self.uvs.data.detach()
            pixel_idx_x = (uvs[:, 0] + 1) / 2 * (atlas_w - 1)
            pixel_idx_x = torch.round(pixel_idx_x).type(torch.int64)
            tile_idx_x = pixel_idx_x // tilew
            tile_pixel_x = pixel_idx_x % tilew
            assert torch.all(torch.logical_or(tile_pixel_x == 0, tile_pixel_x == (tilew - 1)))
            tile_pixel_x[tile_pixel_x == (tilew - 1)] = newtilew - 1
            new_pixel_idx_x = tile_idx_x * newtilew + tile_pixel_x
            new_uvs_x = new_pixel_idx_x / (new_atlas_w - 1) * 2 - 1

            pixel_idx_y = (uvs[:, 1] + 1) / 2 * (atlas_h - 1)
            pixel_idx_y = torch.round(pixel_idx_y).type(torch.int64)
            tile_idx_y = pixel_idx_y // tileh
            tile_pixel_y = pixel_idx_y % tileh
            assert torch.all(torch.logical_or(tile_pixel_y == 0, tile_pixel_y == (tileh - 1)))
            tile_pixel_y[tile_pixel_y == (tileh - 1)] = newtileh - 1
            new_pixel_idx_y = tile_idx_y * newtileh + tile_pixel_y
            new_uvs_y = new_pixel_idx_y / (new_atlas_h - 1) * 2 - 1

            self.uvs.data[:, 0] = new_uvs_x
            self.uvs.data[:, 1] = new_uvs_y

        self.register_parameter("atlas", nn.Parameter(new_atlas, requires_grad=True))
        self.register_parameter("atlas_dyn", nn.Parameter(new_atlas_dyn, requires_grad=True))
        print("MPV.los:: Resizing successful !")

    def get_optimizer(self, step):
        args = self.args
        (_, base_lr), (_, verts_lr) = self.get_lrate(step)

        all_params = {k: v for k, v in self.named_parameters()}
        verts_params_list = ["_verts"]
        base_params_list = set(all_params.keys()) - set(verts_params_list)
        params = [
            {'params': [all_params[k] for k in base_params_list]},  # param_group 0
            {'params': [all_params[k] for k in verts_params_list],  # param_group 1
             'lr': verts_lr}
        ]
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params=params, lr=base_lr, betas=(0.9, 0.999), eps=6e-8)
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

        if 'atlas_dyn' in state_dict.keys():
            atlas_dyn = state_dict['atlas_dyn']
        else:
            atlas_dyn = torch.randn((self.frm_num, 4, self.atlas_full_h, self.atlas_full_w)) * self.args.init_std
        self.atlas_dyn.data = atlas_dyn.type_as(self.atlas)

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
        # ]).type_as(uvs)
        # color = torch.cat([1 - uvs * uv_scaling, torch.zeros_like(uvs[:, :1])], dim=-1)
        color = torch.cat([0.5 - uvs * 0.5, torch.zeros_like(uvs[:, :1])], dim=-1)
        color = np.clip(color.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mesh1 = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(),
                                vertex_colors=color, process=False)
        txt = mesh1.export(prefix + ".obj", "obj")
        with open(prefix + ".obj", 'w') as f:
            f.write(txt)

    @torch.no_grad()
    def save_texture(self, prefix):
        atlas_h, atlas_w = self.atlas.shape[-2:]
        texture = self.atlas.detach()[0].permute(1, 2, 0).reshape(-1, self.args.atlas_cnl)
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

    @property
    def verts(self):
        verts = self._verts
        if self.args.normalize_verts:
            depth_scaling = self.planedepth
            verts = (verts.reshape(len(depth_scaling), -1) * depth_scaling[:, None]).reshape_as(verts)
        return verts

    def render(self, H, W, extrin, intrin, ts):
        F = len(ts)
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

            uvs = uvs.type_as(self.atlas)

        _, ray_direction = get_rays_tensor_batches(H, W, intrin, pose2extrin_torch(extrin))
        ray_direction = ray_direction / ray_direction.norm(dim=-1, keepdim=True)
        ray_direction = ray_direction[..., None, :].expand(pixel_to_face.shape + (3,))
        ray_d = ray_direction.reshape(-1, 3)[mask, :]
        ray_d = self.view_embed_fn(ray_d.reshape(-1, 3)).type_as(self.atlas)

        # uv from 0, S - 1  to -1, 1
        # uv_scaling = torch.tensor([
        #     2 / (self.atlas.shape[-1] - 1),
        #     2 / (self.atlas.shape[-2] - 1)
        # ], device=uvs.device)
        # uvs = uvs * uv_scaling - 1

        HxWxD = len(uvs)
        # dynamic part
        rgba_dyn = torchf.grid_sample(self.atlas_dyn[ts],
                                           uvs[None, None, ...].expand(F, 1, HxWxD, 2),
                                           padding_mode="zeros", align_corners=True)
        rgba_dyn = rgba_dyn.reshape(F, 4, -1).permute(0, 2, 1)

        # static part
        rgba_feat = torchf.grid_sample(self.atlas,
                                       uvs[None, None, ...],
                                       padding_mode="zeros", align_corners=True)
        rgba_feat = rgba_feat.reshape(1, self.atlas.shape[1], -1).permute(0, 2, 1)
        chunksz = self.args.chunk
        tex_input = torch.cat([rgba_feat, ray_d[None]], dim=-1).reshape(HxWxD, -1)
        rgba = torch.cat([self.feat2rgba(tex_input[batchi: batchi + chunksz])
                          for batchi in range(0, len(tex_input), chunksz)])
        rgba = rgba[None]

        # fuse
        rgba = torch.cat([self.rgb_activate(rgba[..., :-1] + rgba_dyn[..., :-1]),
                          self.alpha_activate(rgba[..., -1:] + rgba_dyn[..., -1:])], dim=-1).reshape(F, HxWxD, 4)
        canvas = torch.zeros((F, H, W, self.mpi_d, 4)).type_as(rgba).reshape(F, -1, 4)
        mask_expand = mask[None, :, None].expand_as(canvas)
        mpi = torch.masked_scatter(canvas, mask_expand.expand_as(canvas), rgba)
        mpi = mpi.reshape(F, H, W, self.mpi_d, 4)
        # make rgb d a plane
        rgb, blend_weight = overcompose(
            mpi[..., -1],
            mpi[..., :-1],
        )

        variables = {
            "pix_to_face": pixel_to_face,
            "blend_weight": blend_weight,
            "mpi": mpi,
            "depth": None,  # rgbd[..., -1],
            "alpha": blend_weight.sum(dim=-1)
        }
        return rgb[..., :3], variables

    def forward(self, h, w, tar_extrins, tar_intrins, ts=None, res=None, losscfg=None):
        extrins = tar_extrins @ self.ref_extrin[None, ...].inverse()

        if ts is None:
            ts = torch.arange(self.frm_num).long()
        rgb, variables = self.render(h, w, extrins, tar_intrins, ts)
        frm_num_ori = len(rgb)
        rgb = rgb.permute(0, 3, 1, 2)
        extra = {}
        if self.training:
            assert res is not None
            # main loss
            rgb_pad = rgb
            if self.isloop:
                pad_frame = self.swd_patcht_size - 1
                rgb_pad = torch.cat([rgb, rgb[:pad_frame]], 0)

            losscfg = {k: v[0].item() if torch.is_tensor(v) else v[0] for k, v in losscfg.items()}
            loss_name = losscfg.pop('loss_name')
            loss_gain = losscfg.pop('loss_gain', 1.)
            loss = self.losses[loss_name]
            main_loss = loss(rgb_pad.permute(1, 0, 2, 3)[None],
                             res.permute(0, 2, 1, 3, 4), **losscfg)

            extra['swd'] = main_loss.reshape(1, -1) * loss_gain

            if self.args.sparsity_loss_weight > 0:
                alpha = variables["mpi"][..., -1]
                sparsity = alpha.norm(dim=-1, p=1) / alpha.norm(dim=-1, p=2).clamp_min(1e-4)
                sparsity = sparsity.mean() * loss_gain
                extra["sparsity"] = sparsity.reshape(1, -1)

            if self.args.rgb_smooth_loss_weight > 0:
                smooth = variables["mpi"][..., :-1]
                smoothx = (smooth[:, :, :-1] - smooth[:, :, 1:]).abs().mean()
                smoothy = (smooth[:, :-1] - smooth[:, 1:]).abs().mean()
                smooth = (smoothx + smoothy).reshape(1, -1) * loss_gain
                extra["rgb_smooth"] = smooth.reshape(1, -1)

            if self.args.a_smooth_loss_weight > 0:
                smooth = variables["mpi"][..., -1]
                smoothx = (smooth[:, :, :-1] - smooth[:, :, 1:]).abs().mean()
                smoothy = (smooth[:, :-1] - smooth[:, 1:]).abs().mean()
                smooth = (smoothx + smoothy) * loss_gain
                extra["a_smooth"] = smooth.reshape(1, -1)

            if self.args.density_loss_weight > 0:
                alpha = variables["alpha"]
                density = ((alpha - 1) ** 2).mean() * loss_gain
                extra["density"] = density.reshape(1, -1)

            # if self.args.d_smooth_loss_weight > 0:
            #     depth = variables['depth']
            #     smoothx = (depth[:, :, :-1] - depth[:, :, 1:]).abs().mean()
            #     smoothy = (depth[:, :-1] - depth[:, 1:]).abs().mean()
            #     smooth = (smoothx + smoothy).reshape(1, -1)
            #     extra["d_smooth"] = smooth.reshape(1, -1)

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

            return None, extra

        else:  # if not self.training:
            return rgb, {}
