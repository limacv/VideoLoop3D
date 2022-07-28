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

activate = {'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'none': lambda x: x,
            'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
            'softplus': lambda x: nn.Softplus()(x - 1),
            'tanh': torch.tanh,
            'clamp': lambda x: torch.clamp(x, 0, 1)}


class MPI(nn.Module):
    def __init__(self, args, H, W, ref_extrin, ref_intrin, near, far):
        super(MPI, self).__init__()
        self.args = args
        self.mpi_h, self.mpi_w = int(args.mpi_h_scale * H), int(args.mpi_w_scale * W)
        self.mpi_d, self.near, self.far = args.mpi_d, near, far
        self.H, self.W = H, W
        self.H_start, self.W_start = (self.mpi_h - H) // 2, (self.mpi_w - W) // 2
        assert ref_extrin.shape == (4, 4) and ref_intrin.shape == (3, 3)
        ref_intrin_mpi = get_new_intrin(ref_intrin, - self.H_start, - self.W_start)
        self.register_buffer("ref_extrin", torch.tensor(ref_extrin))
        self.register_buffer("ref_intrin", torch.tensor(ref_intrin_mpi).float())

        planenormal = torch.tensor([0, 0, 1]).reshape(1, 3).repeat(self.mpi_d, 1).float()
        if args.optimize_normal:
            self.register_parameter("plane_normal", nn.Parameter(planenormal, requires_grad=True))
        else:
            self.register_buffer("plane_normal", planenormal)

        planedepth = make_depths(self.mpi_d, near, far).float()
        if args.optimize_depth:
            self.register_parameter("plane_depth", nn.Parameter(planedepth, requires_grad=True))
        else:
            self.register_buffer("plane_depth", planedepth)

        mpi = torch.rand((1, self.mpi_d, 4, self.mpi_h, self.mpi_w))  # RGBA
        mpi[:, :, -1] = -2

        self.register_parameter("mpi", nn.Parameter(mpi, requires_grad=True))
        self.tonemapping = activate['sigmoid']

    def forward(self, h, w, tar_extrins, tar_intrins):
        ref_extrins = self.ref_extrin[None, ...].expand_as(tar_extrins)
        ref_intrins = self.ref_intrin[None, ...].expand_as(tar_intrins)
        homo = compute_homography(ref_extrins, ref_intrins, tar_extrins, tar_intrins,
                                  self.plane_normal[None, ...], self.plane_depth)
        mpi_warp = warp_homography(h, w, homo, self.tonemapping(self.mpi))

        extra = {}
        if self.training:
            if self.args.sparsity_loss_weight > 0:
                sparsity = mpi_warp[:, :, -1].mean()
                extra["sparsity"] = sparsity.reshape(1, -1)
        return overcomposeNto0(mpi_warp), extra


class MPMesh(nn.Module):
    def __init__(self, args, H, W, ref_extrin, ref_intrin, near, far):
        super(MPMesh, self).__init__()
        self.args = args
        self.upsample_stage = args.upsample_stage
        self.mpi_h, self.mpi_w = int(args.mpi_h_scale * H), int(args.mpi_w_scale * W)
        self.mpi_d, self.near, self.far = args.mpi_d, near, far
        self.mpi_h_verts, self.mpi_w_verts = args.mpi_h_verts, args.mpi_w_verts
        self.H, self.W = H, W
        self.atlas_grid_h, self.atlas_grid_w = args.atlas_grid_h, self.mpi_d // args.atlas_grid_h
        self.atlas_size_scale = args.atlas_size_scale
        self.atlas_h = int(self.atlas_grid_h * self.mpi_h * self.atlas_size_scale)
        self.atlas_w = int(self.atlas_grid_w * self.mpi_w * self.atlas_size_scale)

        assert self.mpi_d % self.atlas_grid_h == 0, "mpi_d and atlas_grid_h should match"

        assert ref_extrin.shape == (4, 4) and ref_intrin.shape == (3, 3)
        self.register_buffer("ref_extrin", torch.tensor(ref_extrin))
        self.register_buffer("ref_intrin", torch.tensor(ref_intrin).float())

        # construct the vertices
        planedepth = make_depths(self.mpi_d, near, far).float().flip(0)
        self.register_buffer("planedepth", planedepth)

        # get intrin for mapping entire MPI to image, in order to generate vertices
        self.H_start, self.W_start = (self.mpi_h - H) // 2, (self.mpi_w - W) // 2
        ref_intrin_mpi = get_new_intrin(self.ref_intrin, - self.H_start, - self.W_start)
        verts = torch.meshgrid(
            [torch.linspace(0, self.mpi_h - 1, args.mpi_h_verts), torch.linspace(0, self.mpi_w - 1, args.mpi_w_verts)])
        verts = torch.stack(verts[::-1], dim=-1).reshape(1, -1, 2)
        # num_plane, H*W, 2
        verts = (verts - ref_intrin_mpi[None, None, :2, 2]) * planedepth[:, None, None].type_as(verts)
        verts /= ref_intrin_mpi[None, None, [0, 1], [0, 1]]
        zs = planedepth[:, None, None].expand_as(verts[..., :1])
        verts = torch.cat([verts.reshape(-1, 2), zs.reshape(-1, 1)], dim=-1)
        if args.normalize_verts:
            scaling = self.planedepth
            verts = (verts.reshape(len(scaling), -1) / scaling[:, None]).reshape_as(verts)

        uvs_plane = torch.meshgrid([torch.arange(self.atlas_grid_h) / self.atlas_grid_h,
                                    torch.arange(self.atlas_grid_w) / self.atlas_grid_w])
        uvs_plane = torch.stack(uvs_plane[::-1], dim=-1) * 2 - 1
        uvs_voxel_size = (- uvs_plane[-1, -1] + 1).reshape(1, 1, 2)
        uvs_voxel = torch.meshgrid([torch.linspace(0, 1, args.mpi_h_verts), torch.linspace(0, 1, args.mpi_w_verts)])
        uvs_voxel = torch.stack(uvs_voxel[::-1], dim=-1).reshape(1, -1, 2) * uvs_voxel_size
        uvs = (uvs_plane.reshape(-1, 1, 2) + uvs_voxel.reshape(1, -1, 2)).reshape(-1, 2)

        verts_indice = torch.arange(len(verts)).reshape(self.mpi_d, args.mpi_h_verts, args.mpi_w_verts)
        faces013 = torch.stack([verts_indice[:, :-1, :-1], verts_indice[:, 1:, :-1], verts_indice[:, 1:, 1:]], -1)
        faces320 = torch.stack([verts_indice[:, 1:, 1:], verts_indice[:, :-1, 1:], verts_indice[:, :-1, :-1]], -1)
        faces = torch.cat([faces013.reshape(-1, 3), faces320.reshape(-1, 3)])

        scaling = 0.5 ** len(self.upsample_stage)
        atlas = torch.rand((1, args.atlas_cnl, int(self.atlas_h * scaling), int(self.atlas_w * scaling)))

        # -1, 1 to 0, h
        uvs = uvs * 0.5 + 0.5
        atlas_size = torch.tensor([int(self.atlas_w * scaling), int(self.atlas_h * scaling)]).reshape(-1, 2)
        uvs *= (atlas_size - 1).type_as(uvs)

        self.register_parameter("uvs", nn.Parameter(uvs, requires_grad=True))
        self._verts = nn.Parameter(verts, requires_grad=True)
        self.register_buffer("faces", faces.long())
        self.optimize_geometry = False
        self.register_parameter("atlas", nn.Parameter(atlas, requires_grad=True))

        self.view_embed_fn, self.view_cnl = get_embedder(args.multires_views, input_dim=3)
        if args.rgb_mlp_type == "direct":
            self.feat2rgba = lambda x: x[..., :4]
            atlas[:, -1] = -2
            self.use_viewdirs = False
        elif args.rgb_mlp_type == "rgbamlp":
            self.feat2rgba = nn.Sequential(
                nn.Linear(self.view_cnl + args.atlas_cnl, 48), nn.ReLU(),
                nn.Linear(48, 4)
            )
            self.feat2rgba[-2].bias.data[-1] = -2
            self.use_viewdirs = True
        elif args.rgb_mlp_type == "rgbmlp":
            self.feat2rgba = Feat2RGBMLP_alpha(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
            atlas[:, 0] = -2
        elif args.rgb_mlp_type == "rgbanex":
            self.feat2rgba = NeX_RGBA(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
        elif args.rgb_mlp_type == "rgbnex":
            self.feat2rgba = NeX_RGB(args.atlas_cnl, self.view_cnl)
            self.use_viewdirs = True
            atlas[:, 0] = -2
        else:
            raise RuntimeError(f"rgbmlp_type = {args.rgb_mlp_type} not recognized")
        self.rgb_activate = activate[args.rgb_activate]

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
        if step in self.upsample_stage:
            scaling = 0.5 ** (len(self.upsample_stage) - self.upsample_stage.index(step) - 1)
            scaled_size = int(self.atlas_h * scaling), int(self.atlas_w * scaling)
            print(f"  Upsample to {scaled_size} in step {step}")
            self.register_parameter("atlas",
                                    nn.Parameter(
                                        torchf.upsample(self.atlas, scaled_size, mode='bilinear'),
                                        requires_grad=True))
            with torch.no_grad():
                uv_scaling = torch.tensor([
                    (scaled_size[1] - 1) / (self.atlas.shape[-1] - 1),
                    (scaled_size[0] - 1) / (self.atlas.shape[-2] - 1),
                ]).reshape(-1, 2).type_as(self.uvs)
                self.uvs *= uv_scaling

    # def post_backward(self):
    # if self.verts.grad is not None:
    #     # the grad_graph is scale with uv, so we redo the scaling
    #     uv_scaling = max(self.atlas.shape)
    #     depth_scaling = self.planedepth / self.planedepth[0]
    #     graddata = self.verts.grad.data.reshape(self.mpi_d, -1)
    #     graddata *= (self.args.optimize_verts_gain / uv_scaling * depth_scaling[:, None])
    # if self.uvs.grad is not None:
    #     self.uvs.grad.data *= self.args.optimize_uvs_gain

    def save_mesh(self, prefix):
        vertices, faces, uvs = self.verts.detach(), self.faces.detach(), self.uvs.detach()
        uv_scaling = torch.tensor([
            1 / (self.atlas.shape[-1] - 1),
            1 / (self.atlas.shape[-2] - 1)
        ])
        color = torch.cat([1 - uvs * uv_scaling, torch.zeros_like(uvs[:, :1])], dim=-1)
        color = np.clip(color.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mesh1 = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(),
                                vertex_colors=color)
        txt = mesh1.export(prefix + ".obj", "obj")
        with open(prefix + ".obj", 'w') as f:
            f.write(txt)

    @torch.no_grad()
    def save_texture(self, prefix):
        texture = self.atlas.detach()[0].permute(1, 2, 0).reshape(-1, self.args.atlas_cnl)
        ray_dir = torch.tensor([[0, 0, 1.]]).type_as(texture).expand(len(texture), -1)
        ray_dir = self.view_embed_fn(ray_dir[:, :2])
        tex_input = torch.cat([texture, ray_dir], dim=-1)
        chunksz = self.args.chunk
        rgba = torch.cat([self.feat2rgba(tex_input[batchi: batchi + chunksz])
                          for batchi in range(0, len(tex_input), chunksz)])
        rgba = self.rgb_activate(rgba)
        texture = (rgba * 255).type(torch.uint8).reshape(self.atlas_h, self.atlas_w, 4).cpu().numpy()
        import imageio
        imageio.imwrite(prefix + ".png", texture)

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
            vertices_index = faces[0, faces_ma]
            uvs = self.uvs[vertices_index]  # N, 3, n_feat
            bary_coords_ma = bary_coords.reshape(-1, 3)[mask, :]  # N, 3
            uvs = (bary_coords_ma[..., None] * uvs).sum(dim=-2)

        _, ray_direction = get_rays_tensor_batches(H, W, intrin, pose2extrin_torch(extrin))
        ray_direction = ray_direction / ray_direction.norm(dim=-1, keepdim=True)
        ray_direction = ray_direction[..., None, :].expand(pixel_to_face.shape + (3,))
        ray_d = ray_direction.reshape(-1, 3)[mask, :]
        ray_d = self.view_embed_fn(ray_d.reshape(-1, 3)[:, :2])  # only use xy for ray direction

        # uv from 0, S - 1  to -1, 1
        uv_scaling = torch.tensor([
            2 / (self.atlas.shape[-1] - 1),
            2 / (self.atlas.shape[-2] - 1)
        ])
        uvs = uvs * uv_scaling - 1
        rgba_feat = torchf.grid_sample(self.atlas,
                                       uvs[None, None, ...],
                                       padding_mode="zeros")
        rgba_feat = rgba_feat.reshape(self.atlas.shape[1], -1).permute(1, 0)
        chunksz = self.args.chunk
        tex_input = torch.cat([rgba_feat, ray_d], dim=-1)
        rgba = torch.cat([self.feat2rgba(tex_input[batchi: batchi + chunksz])
                          for batchi in range(0, len(tex_input), chunksz)])
        rgba = self.rgb_activate(rgba)
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
                sparsity = variables["mpi"][..., -1].abs().mean()
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
