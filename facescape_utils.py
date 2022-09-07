"""
This script can be used as a rendering library
"""
from concurrent.futures import thread
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchf
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    rasterize_meshes,
    RasterizationSettings,
    TexturesVertex,
    TexturesUV
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
import matplotlib.pyplot as plt


def to_batched_tensor(a):
    a = torch.tensor(a)[None].cuda()
    if a.dtype in (torch.float16, torch.float32, torch.float64):
        return a.float()
    elif a.dtype in (torch.int16, torch.int32, torch.int64):
        return a.long()
    else:
        return a


def unifying_batchdim(*args):
    batch_sz = max(a.shape[0] for a in args)
    return [a.expand(batch_sz, *a.shape[1:]) for a in args]


def get_uv_verts(uvs):
    """generate vertices in texture space based on uv coordinate"""
    uvs = uvs * 2 - 1
    return torch.stack([uvs[..., 0], -uvs[..., 1], torch.ones_like(uvs[..., 0])], -1)


def proj_img_to_texture(tH, tW, verts, faces, poses, intrins, observations, uvs, faceuvs=None,
                        ):
    """
    observations: N x H x W x C
    uvs: uv in [0, 1], with u+ left->right v+ bottom->up
    currently observations/poses/intrins can be batched naively
    """
    vert_new = get_uv_verts(uvs)
    face_new = faceuvs if faceuvs is not None else faces

    # TODO: batchfy executing the following:
    h, w, _ = observations.shape[-3:]
    results = []
    for pose, intrin, obs in zip(poses[:, None], intrins[:, None], observations[:, None]):
        R = pose[:, :3, :3].inverse()
        T = - (R @ pose[:, :, 3:]).reshape(-1, 3)
        vert_view = (R[:, None] @ verts[..., None] + T[:, None, :, None])
        vert_ndc = (intrin[:, None, :3, :3] @ vert_view)[..., 0]
        vert_ndc = vert_ndc[..., :2] / vert_ndc[..., 2:]
        vert_ndc = vert_ndc / torch.tensor([[[w, h]]]).type_as(verts)
        uv_new = torch.stack([vert_ndc[..., 0], 1 - vert_ndc[..., 1]], dim=-1)

        res = render_uvtex_image(tH, tW, vert_new, face_new, obs, uv_new, faces)
        results.append(res)

    return torch.cat(results)


def proj_img_to_vert(verts, poses, intrins, observations):
    """
    observations: N x H x W x C
    currently observations/poses/intrins can be batched naively
    """
    h, w, _ = observations.shape[-3:]

    R = poses[:, :3, :3].inverse()
    T = - (R @ poses[:, :, 3:]).reshape(-1, 3)
    vert_view = (R[:, None] @ verts[..., None] + T[:, None, :, None])
    vert_ndc = (intrins[:, None, :3, :3] @ vert_view)[..., 0]
    vert_ndc = vert_ndc[..., :2] / vert_ndc[..., 2:]
    uv = vert_ndc * torch.tensor([[[2 / w, 2 / h]]]).type_as(verts) - 1

    vertattri = torchf.grid_sample(observations.permute(0, 3, 1, 2), uv[:, None], align_corners=True)
    return vertattri[..., 0, :].permute(0, 2, 1)


def get_occlusion_texture(tH, tW, verts, faces,
                          H, W, poses, intrins, uvs, faceuvs=None, threshold=0.01):
    """
    generating occlusion maps when observing from poses and intrins. The map is in texture space
    tH, tW: the generated shadow texture size
    H, W: the observation size
    uvs: uv in [0, 1], with u+ left->right v+ bottom->up
    """
    # first render depth map
    frag = rasterize(
        H, W,
        verts, faces,
        poses, intrins
    )
    depth_map = frag.zbuf
    depth_map_proj = proj_img_to_texture(tH, tW, verts, faces, poses, intrins, depth_map, uvs, faceuvs)

    # then get depth vertices
    R = poses[:, :3, :3].inverse()
    T = - (R @ poses[:, :, 3:]).reshape(-1, 3)
    vert_view = (R[:, None] @ verts[..., None] + T[:, None, :, None])
    vert_depth = vert_view[..., -1:, 0]

    vert_new = get_uv_verts(uvs)
    face_new = faceuvs if faceuvs is not None else faces

    depth_proj = [render_verttex_image(tH, tW, vert_new, face_new, vd) for vd in vert_depth[:, None]]
    depth_proj = torch.cat(depth_proj)[..., 0]  # ..., face_per_pixel, feature_dim
    return depth_map_proj[..., 0] + threshold > depth_proj


def get_occlusion_vertattri(verts, faces,
                            H, W, poses, intrins, threshold=0.01):
    """
    generating occlusion maps when observing from poses and intrins. The map is in texture space
    tH, tW: the generated shadow texture size
    H, W: the observation size
    uvs: uv in [0, 1], with u+ left->right v+ bottom->up
    """
    # first render depth map
    frag = rasterize(
        H, W,
        verts, faces,
        poses, intrins
    )
    depth_map = frag.zbuf
    depth_map_proj = proj_img_to_vert(verts, poses, intrins, depth_map)

    # then get depth vertices
    R = poses[:, :3, :3].inverse()
    T = - (R @ poses[:, :, 3:]).reshape(-1, 3)
    vert_view = (R[:, None] @ verts[..., None] + T[:, None, :, None])
    vert_depth = vert_view[..., -1:, 0]
    return depth_map_proj + threshold > vert_depth


def render_uvtex_image(H, W, verts, faces, textures, uvs, faceuvs=None, poses=None, intrins=None,
                       ret_fragment=False, cull_backface=False):
    """
    uv in [0, 1], with u+ left->right, v+ bottom->up
    The align_corner=True, indicates the uv is usually get by x_im / w, y_im / h
    """
    frag = rasterize(
        H, W,
        verts, faces,
        poses, intrins,
        cull_backface
    )
    if faceuvs is None:
        faceuvs = faces

    texture_pt3d = TexturesUV(textures, faceuvs, uvs, padding_mode='zeros')
    result = texture_pt3d.sample_textures(frag)[..., 0, :]

    if ret_fragment:
        return result, frag
    return result


def render_verttex_image(H, W, verts, faces, vert_attrs, poses=None, intrins=None,
                         ret_fragment=False, cull_backface=False):
    """currently dont support batch operation"""
    frag = rasterize(
        H, W,
        verts, faces,
        poses, intrins,
        cull_backface
    )
    verttex = TexturesVertex(vert_attrs)
    result = verttex.sample_textures(frag, faces[0])[..., 0, :]

    if ret_fragment:
        return result, frag
    return result


def rasterize(H, W, verts, faces, poses=None, intrins=None, cull_backface=False) -> Fragments:
    """
    support batched operation
    if poses and intrins is None, the verts should be in ndc space, i.e.: x, y ~ [-1, 1] z ~ (0, 9999)
    """
    if poses is not None:
        R = poses[:, :3, :3].inverse()
        T = - (R @ poses[:, :, 3:]).reshape(-1, 3)
        vert_view = (R[:, None] @ verts[..., None] + T[:, None, :, None])
    else:
        vert_view = verts

    if intrins is not None:
        vert_ndc = (intrins[:, None, :3, :3] @ vert_view)[..., 0]
        vert_ndc = vert_ndc[..., :2] / vert_ndc[..., 2:]
        vert_ndc = vert_ndc / torch.tensor([[[W / 2, H / 2]]]).type_as(verts) - 1
        verts_render = torch.cat([vert_ndc[..., :2], vert_view[..., 2:3, 0]], dim=-1)
    else:
        verts_render = vert_view

    if H < W:  # strange trick to make raster result correct
        scaling = torch.tensor([[[-W / H, -1, 1]]])
        # verts[..., 0] = verts[..., 0] * (-W / H)
        # verts[..., 1] = verts[..., 1] * -1
    else:
        scaling = torch.tensor([[[-1, -H / W, 1]]])
        # verts[..., 0] = verts[..., 0] * -1
        # verts[..., 1] = verts[..., 1] * (-H / W)
    verts_render = verts_render * scaling.type_as(verts_render)

    raster_settings = RasterizationSettings(
        image_size=(H, W),  # viewport
        blur_radius=0,
        max_faces_per_bin=len(faces[0]) // 2,
        faces_per_pixel=1,
        cull_backfaces=cull_backface
    )
    raster = SimpleRasterizer(raster_settings)
    verts_render, faces = unifying_batchdim(verts_render, faces)
    frag = raster(verts_render, faces)
    return frag


# mesh related
class SimpleRasterizer(nn.Module):
    def __init__(self, raster_settings=None):
        """
        max_faces_per_bin = int(max(10000, meshes._F / 5))
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()
        self.raster_settings = raster_settings

    def forward(self, vertices, faces):
        """
        Args:
            vertices: B, N, 3
            faces: B, N, 3
        """
        raster_settings = self.raster_settings

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        perspective_correct = True
        z_clip = raster_settings.z_clip_value
        # z_clip should be set to >0 value if there are some meshes comming near the camera

        fragment = rasterize_meshes(
            Meshes(vertices, faces),
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            clip_barycentric_coords=clip_barycentric_coords,
            perspective_correct=perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
            z_clip_value=z_clip,
            cull_to_frustum=raster_settings.cull_to_frustum,
        )
        return Fragments(*fragment)  # pix_to_face, zbuf, bary_coords, dists

