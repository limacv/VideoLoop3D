import torch
import torch.nn as nn
import torch.nn.functional as torchf
import os
import imageio
import time
import cv2
from utils import *
from utils_mpi import *
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    rasterize_meshes,
    RasterizationSettings,
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
        self.mpi_h, self.mpi_w = int(args.mpi_h_scale * H), int(args.mpi_w_scale * W)
        self.mpi_d, self.near, self.far = args.mpi_d, near, far
        self.mpi_h_verts, self.mpi_w_verts = args.mpi_h_verts, args.mpi_w_verts
        self.H, self.W = H, W
        self.atlas_grid_h, self.atlas_grid_w = args.atlas_grid_h, self.mpi_d // args.atlas_grid_h
        self.atlas_size_scale = args.atlas_size_scale
        self.atlas_h = int(self.atlas_grid_h * H * self.atlas_size_scale)
        self.atlas_w = int(self.atlas_grid_w * W * self.atlas_size_scale)

        assert self.mpi_d % self.atlas_grid_h == 0, "mpi_d and atlas_grid_h should match"

        assert ref_extrin.shape == (4, 4) and ref_intrin.shape == (3, 3)
        self.register_buffer("ref_extrin", torch.tensor(ref_extrin))
        self.register_buffer("ref_intrin", torch.tensor(ref_intrin).float())

        # construct the vertices
        planedepth = make_depths(self.mpi_d, near, far).float().flip(0)
        # TODO: add margin
        verts = torch.meshgrid([torch.linspace(0, H - 1, args.mpi_h_verts), torch.linspace(0, W - 1, args.mpi_w_verts)])
        verts = torch.stack(verts[::-1], dim=-1).reshape(1, -1, 2)
        # num_plane, H*W, 2
        verts = (verts - self.ref_intrin[None, None, :2, 2]) * planedepth[:, None, None].type_as(verts)
        verts /= self.ref_intrin[None, None, [0, 1], [0, 1]]
        zs = planedepth[:, None, None].expand_as(verts[..., :1])
        verts = torch.cat([verts.reshape(-1, 2), zs.reshape(-1, 1)], dim=-1)

        uvs_plane = torch.meshgrid([torch.arange(self.atlas_grid_h) / self.atlas_grid_h,
                                    torch.arange(self.atlas_grid_w) / self.atlas_grid_w])
        uvs_plane = torch.stack(uvs_plane[::-1], dim=-1) * 2 - 1
        uvs_voxel_size = (- uvs_plane[-1, -1] + 1).reshape(1, 1, 2)
        uvs_voxel = torch.meshgrid([torch.linspace(0, 1, args.mpi_h_verts), torch.linspace(0, 1, args.mpi_w_verts)])
        uvs_voxel = torch.stack(uvs_voxel[::-1], dim=-1).reshape(1, -1, 2) * uvs_voxel_size
        uvs = (uvs_plane.reshape(-1, 1, 2) + uvs_voxel.reshape(1, -1, 2)).reshape(-1, 2)

        verts_indice = torch.arange(len(verts)).reshape(self.mpi_d, args.mpi_h_verts, args.mpi_w_verts)
        faces013 = torch.stack([verts_indice[:, :-1, :-1], verts_indice[:, 1:, :-1], verts_indice[:, 1:, 1:]], -1)
        faces023 = torch.stack([verts_indice[:, :-1, 1:], verts_indice[:, :-1, :-1], verts_indice[:, 1:, 1:]], -1)
        faces = torch.cat([faces013.reshape(-1, 3), faces023.reshape(-1, 3)])

        self.register_parameter("uvs", nn.Parameter(torch.tensor(uvs), requires_grad=True))
        self.register_parameter("verts", nn.Parameter(torch.tensor(verts), requires_grad=True))
        self.register_buffer("faces", faces.long())

        self.save_mesh("M:\\VideoLoops\\mpi_mesh")

        atlas = torch.rand((1, 4, self.atlas_h, self.atlas_w))
        atlas[:, -1] = -2
        self.register_parameter("atlas", nn.Parameter(atlas, requires_grad=True))
        self.tonemapping = activate['sigmoid']

    def save_mesh(self, prefix):
        vertices, faces, uvs = self.verts.detach(), self.faces.detach(), self.uvs.detach()
        color = torch.cat([(uvs + 1) / 2, torch.zeros_like(uvs[:, :1])], dim=-1)
        color = np.clip(color.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        mesh1 = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy(),
                                vertex_colors=color)
        txt = mesh1.export(prefix + ".obj", "obj")
        with open(prefix + ".obj", 'w') as f:
            f.write(txt)

    def render(self, H, W, extrin, intrin):
        B = len(extrin)
        verts, faces = self.verts.reshape(1, -1, 3), self.faces.reshape(1, -1, 3)

        R, T = extrin[:, :3, :3], extrin[:, :3, 3]
        # normalize intrin to ndc
        intrin = intrin.clone()
        if H < W:  # strange trick to make raster result correct
            intrin[:, :2] *= (- 2 / H)
            intrin[:, 0, 2] += W / H
            intrin[:, 1, 2] += 1
        else:
            intrin[:, :2] *= (- 2 / W)
            intrin[:, 0, 2] += 1
            intrin[:, 1, 2] += H / W

        # transform to ndc space
        vert_view = (R @ verts[..., None] + T[..., None])
        vert_ndc = (intrin[:, :3, :3] @ vert_view)[..., 0]
        vert_ndc = vert_ndc[..., :2] / vert_ndc[..., 2:]
        vert = torch.cat([vert_ndc[..., :2], vert_view[..., 2:3, 0]], dim=-1)

        # rasterize
        raster_settings = RasterizationSettings(
            image_size=(H, W),  # viewport
            blur_radius=0.0,
            faces_per_pixel=self.mpi_d,
        )
        raster = SimpleRasterizer(raster_settings)
        pixel_to_face, zbuf, bary_coords, dists = raster(
            vert, faces
        )

        # currently the batching is not supported
        mask = pixel_to_face.reshape(-1) >= 0
        faces_ma = pixel_to_face.reshape(-1)[mask]
        vertices_index = faces[0, faces_ma]
        uvs = self.uvs[vertices_index]  # N, 3, n_feat
        bary_coords_ma = bary_coords.reshape(-1, 3)[mask, :]  # N, 3
        uvs = (bary_coords_ma[..., None] * uvs).sum(dim=-2)
        # TODO: add view dependency

        rgba = torchf.grid_sample(self.atlas,
                                  uvs[None, None, ...],
                                  padding_mode="zeros")
        rgba = rgba.reshape(4, -1).permute(1, 0)
        rgba = self.tonemapping(rgba)

        canvas = torch.zeros((B, H, W, self.mpi_d, 4)).type_as(rgba).reshape(-1, 4)
        mpi = torch.masked_scatter(canvas, mask[:, None].expand_as(canvas), rgba)
        mpi = mpi.reshape(B, H, W, self.mpi_d, 4)
        rgb, blend_weight = overcompose(mpi)

        variables = {
            "blend_weight": blend_weight,
            "mpi": mpi,
            "alpha": blend_weight.sum(dim=-1)
        }
        return rgb, variables

    def forward(self, h, w, tar_extrins, tar_intrins):
        extrins = tar_extrins @ self.ref_extrin[None, ...].inverse()

        rgb, variables = self.render(h, w, extrins, tar_intrins)

        extra = {}
        if self.training:
            if self.args.sparsity_loss_weight > 0:
                sparsity = variables["mpi"][..., -1].abs().mean()
                extra["sparsity"] = sparsity.reshape(1, -1)
        return rgb, extra


class SimpleVideo(nn.Module):
    def __init__(self, args, H, W, time_len, ref_intrin):
        super(SimpleVideo, self).__init__()
        self.args = args
        self.h, self.w = H, W
        self.time_len = time_len
        video = torch.rand((time_len, 3, self.mpi_h, self.mpi_w))
        self.register_parameter("video", nn.Parameter(video, requires_grad=True))

    def forward(self, h, w, tar_extrins, tar_intrins):
        with torch.no_grad():
            homo = compute_homography(ref_extrins, ref_intrins, tar_extrins, tar_intrins,
                                      self.plane_normal, self.plane_depth)
        mpi_warp = warp_homography(h, w, homo, self.tonemapping(self.mpi))

        extra = {}
        if self.training:
            if self.args.sparsity_loss_weight > 0:
                sparsity = mpi_warp[:, :, -1].mean()
                extra["sparsity"] = sparsity.reshape(1, -1)
        return overcomposeNto0(mpi_warp), extra

