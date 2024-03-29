import os
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
from pytorch3d.structures import Meshes
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.renderer.mesh.rasterizer import Fragments
import cv2


img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.tensor([10.]).type_as(x))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


class SimpleRasterizer(nn.Module):
    def __init__(self, raster_settings=None, adaptive_layernum=True):
        """
        max_faces_per_bin = int(max(10000, meshes._F / 5))
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()
        self.raster_settings = raster_settings
        self.adaptive_layer_num = adaptive_layernum

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
        if self.adaptive_layer_num:
            with torch.no_grad():
                pix_to_face = fragment[0]
                pix_to_face_max = pix_to_face.reshape(-1, pix_to_face.shape[-1]).max(dim=0)[0]
                num_layer = torch.count_nonzero(pix_to_face_max > 0).item()
            fragment = [frag[:, :, :, :num_layer] for frag in fragment]
        return Fragments(*fragment)  # pix_to_face, zbuf, bary_coords, dists


def frag2uv(frag: Fragments, uvs, uvfaces):
    """
    return MPI mask, uv coordinate
    """
    pixel_to_face, depths, bary_coords = frag.pix_to_face, frag.zbuf, frag.bary_coords
    # currently the batching is not supported
    mask = pixel_to_face.reshape(-1) >= 0
    mask_flat = mask.reshape(-1)
    faces_ma_dyn = pixel_to_face.reshape(-1)[mask_flat]
    uv_indices = uvfaces[faces_ma_dyn]
    uvs = uvs[uv_indices]  # N, 3, n_feat
    bary_coords_ma = bary_coords.reshape(-1, 3)[mask_flat, :]  # N, 3
    uvs = (bary_coords_ma[..., None] * uvs).sum(dim=-2)
    return mask, uvs


class ParamsWithGradGain(nn.Module):
    def __init__(self, param, grad_gain=1.):
        super(ParamsWithGradGain, self).__init__()
        if grad_gain == 0:
            self.register_buffer("param", param)
        else:
            self.register_parameter("param", nn.Parameter(param, requires_grad=True))

        def grad_gain_fn(grad):
            return grad * grad_gain

        if grad_gain != 1:
            self.param.register_hook(grad_gain_fn)

    def forward(self):
        return self.param


# class InputAdaptor(nn.Module):
#     def __init__(self, H, W):
#         super().__init__()
#         flows = torch
#
#     def forward(self, rgb):


def generate_patchinfo(H_, W_, patch_size_, patch_stride_):
    patch_h_size, patch_w_size = patch_size_
    patch_h_stride, patch_w_stride = patch_stride_

    # generate patch information
    patch_h_start = np.arange(0, H_ - patch_h_size + patch_h_stride, patch_h_stride)
    patch_w_start = np.arange(0, W_ - patch_w_size + patch_w_stride, patch_w_stride)

    patch_wh_start = np.meshgrid(patch_h_start, patch_w_start)
    patch_wh_start = np.stack(patch_wh_start[::-1], axis=-1).reshape(-1, 2)[None, ...]

    patch_wh_start = patch_wh_start.reshape(-1, 2)
    patch_wh_start = torch.tensor(patch_wh_start)

    H_pad = patch_h_start.max() + patch_h_size - H_
    W_pad = patch_w_start.max() + patch_w_size - W_
    assert patch_h_stride > H_pad >= 0 and patch_w_stride > W_pad >= 0, "bug occurs!"

    pad_info = [0, W_pad, 0, H_pad]
    return patch_wh_start, pad_info


def xyz2uv_stereographic(xyz: torch.Tensor, normalized=False):
    """
    xyz: tensor of size (B, 3)
    """
    if not normalized:
        xyz = xyz / xyz.norm(dim=-1, keepdim=True)
    x, y, z = torch.split(xyz, 1, dim=-1)
    z = torch.clamp_max(z, 0.99)
    denorm = torch.reciprocal(-z + 1)
    u, v = x * denorm, y * denorm
    return torch.cat([u, v], dim=-1)


def uv2xyz_stereographic(uv: torch.Tensor):
    u, v = torch.split(uv, 1, dim=-1)
    u2v2 = u ** 2 + v ** 2
    x = u * 2 / (u2v2 + 1)
    y = v * 2 / (u2v2 + 1)
    z = (u2v2 - 1) / (u2v2 + 1)
    return torch.cat([x, y, z], dim=-1)


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    pixelpoints = np.stack([i, j, np.ones_like(i)], -1)[..., np.newaxis]
    localpoints = np.linalg.inv(K) @ pixelpoints

    rays_d = (c2w[:3, :3] @ localpoints)[..., 0]
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_tensor(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()

    pixelpoints = torch.stack([i, j, torch.ones_like(i)], -1).unsqueeze(-1)
    localpoints = torch.matmul(torch.inverse(K), pixelpoints)

    rays_d = torch.matmul(c2w[:3, :3], localpoints)[..., 0]
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_tensor_batches(H, W, K, c2w):
    i, j = torch.meshgrid([torch.linspace(0, W - 1, W, device=K.device),
                          torch.linspace(0, H - 1, H, device=K.device)])
    i = i.t()
    j = j.t()

    pixelpoints = torch.stack([i, j, torch.ones_like(i)], -1)[None, ..., None]
    localpoints = torch.matmul(torch.inverse(K)[:, None, None], pixelpoints)

    rays_d = torch.matmul(c2w[:, None, None, :3, :3], localpoints)[..., 0]
    rays_o = c2w[:, :3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_new_intrin(old_intrin, new_h_start, new_w_start):
    new_intrin = old_intrin.clone() if isinstance(old_intrin, torch.Tensor) else old_intrin.copy()
    new_intrin[..., 0, 2] -= new_w_start
    new_intrin[..., 1, 2] -= new_h_start
    return new_intrin


def pose2extrin_np(pose: np.ndarray):
    if pose.shape[-2] == 3:
        bottom = pose[..., :1, :].copy()
        bottom[..., :] = [0, 0, 0, 1]
        pose = np.concatenate([pose, bottom], axis=-2)
    return np.linalg.inv(pose)


def pose2extrin_torch(pose):
    """
    pose to extrin or extrin to pose (equivalent)
    """
    if pose.shape[-2] == 3:
        bottom = pose[..., :1, :].detach().clone()
        bottom[..., :] = torch.tensor([0, 0, 0, 1.])
        pose = torch.cat([pose, bottom], dim=-2)
    return torch.inverse(pose)


def raw2poses(rot_raw, tran_raw, intrin_raw):
    x = rot_raw[..., 0]
    x = x / torch.norm(x, dim=-1, keepdim=True)
    z = torch.cross(x, rot_raw[..., 1])
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x)
    rot = torch.stack([x, y, z], dim=-1)
    pose = torch.cat([rot, tran_raw[..., None]], dim=-1)
    bottom = torch.tensor([0, 0, 1]).type_as(intrin_raw).reshape(-1, 1, 3).expand(len(intrin_raw), -1, -1)
    intrinsic = torch.cat([intrin_raw, bottom], dim=1)
    return pose, intrinsic


def get_batched_rays_tensor(H, W, Ks, c2ws):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()
    j = j.t()

    pixelpoints = torch.stack([i, j, torch.ones_like(i)], -1)[None, ..., None]
    localpoints = torch.matmul(torch.inverse(Ks)[:, None, None, ...], pixelpoints)

    rays_d = torch.matmul(c2ws[:, None, None, :3, :3], localpoints)[..., 0]
    rays_o = c2ws[:, None, None, :3, -1].expand(rays_d.shape)
    return torch.stack([rays_o, rays_d], dim=1)


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def gaussian(img, kernel_size):
    return GaussianBlur(kernel_size)(img)


def dilate(alpha: torch.Tensor, kernelsz=3, dilate=1):
    """
    alpha: B x L x H x W
    """
    padding = (dilate * (kernelsz - 1) + 1) // 2
    batchsz, layernum, hei, wid = alpha.shape
    alphaunfold = torch.nn.Unfold(kernelsz, dilation=dilate, padding=padding, stride=1)(alpha.reshape(-1, 1, hei, wid))
    alphaunfold = alphaunfold.max(dim=1)[0]
    return alphaunfold.reshape_as(alpha)


def erode(alpha: torch.Tensor, kernelsz=3, dilate=1):
    """
    alpha: B x L x H x W
    """
    padding = (dilate * (kernelsz - 1) + 1) // 2
    batchsz, layernum, hei, wid = alpha.shape
    alphaunfold = torch.nn.Unfold(kernelsz, dilation=dilate, padding=padding, stride=1)(alpha.reshape(-1, 1, hei, wid))
    alphaunfold = alphaunfold.min(dim=1)[0]
    return alphaunfold.reshape_as(alpha)


class DataParallelCPU:
    def __init__(self, module: nn.Module):
        self.module = module

    def to(self, device):
        self.module.to(device)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def compute_loopable_mask(vid, eps=15 / 255, factor=2):
    ori_size = vid[0].shape[:2]

    vid0 = cv2.resize(vid[0], None, None, 1 / factor, 1 / factor)
    rises = np.zeros_like(vid0) > 0
    falls = np.zeros_like(vid0) > 0
    minval = vid0
    maxval = vid0
    for im in vid[1:]:
        im_down = cv2.resize(im, None, None, 1 / factor, 1 / factor)
        minval = np.minimum(minval, im_down)
        maxval = np.maximum(maxval, im_down)
        rises = np.logical_or(im_down - minval > eps, rises)
        falls = np.logical_or(maxval - im_down > eps, falls)

    unchangging = np.logical_and(np.logical_not(rises), np.logical_not(falls))
    unchangging = np.all(unchangging, axis=-1)
    unloopable = np.logical_xor(rises, falls)
    unloopable = np.any(unloopable, axis=-1)
    loopable = np.logical_not(np.logical_or(unchangging, unloopable))

    # loopable = cv2.erode(loopable.astype(np.uint8), np.ones((3, 3)))
    # loopable = cv2.dilate(loopable.astype(np.uint8), np.ones((3, 3)))
    label = np.stack([loopable, unloopable.astype(np.uint8), unchangging.astype(np.uint8)], axis=-1) * 255
    label_smooth = cv2.GaussianBlur(label, (5, 5), 0)
    label_smooth = cv2.resize(label_smooth.astype(np.float32), ori_size[::-1], None)
    loopable_smooth = label_smooth.argmax(axis=-1) == 0
    return loopable_smooth


def save_obj_multimaterial(file, vertices, faces_list, uvs, uvfaces_list, mtls_list):
    with open(file, 'w') as f:
        for vertice in vertices:
            f.write(f"v {vertice[0]} {vertice[1]} {vertice[2]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        for mtl, faces, uvfaces in zip(mtls_list, faces_list, uvfaces_list):
            faces1 = faces + 1
            uvfaces1 = uvfaces + 1
            f.write(f"usemtl {mtl}\n")
            f.write(f"s off\n")
            for face, uvface in zip(faces1, uvfaces1):
                f.write(f"f {face[0]}/{uvface[0]} {face[1]}/{uvface[1]} {face[2]}/{uvface[2]}\n")

        f.write("\n")


def save_obj_with_vcolor(file, verts_colors, faces, uvs, uvfaces):
    with open(file, 'w') as f:
        for pos_color in verts_colors:
            f.write(f"v {pos_color[0]} {pos_color[1]} {pos_color[2]} {pos_color[3]} {pos_color[4]} {pos_color[5]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        faces1 = faces + 1
        uvfaces1 = uvfaces + 1
        for face, uvface in zip(faces1, uvfaces1):
            f.write(f"f {face[0]}/{uvface[0]} {face[1]}/{uvface[1]} {face[2]}/{uvface[2]}\n")

        f.write("\n")


# Mesh utility


def normalize_uv(uv, h, w):
    uv[:, 1] = -uv[:, 1]
    uv = uv * 0.5 + 0.5
    uv = uv * np.array([w - 1, h - 1]) / np.array([w, h]) + 0.5 / np.array([w, h])
    return uv


def cull_unused(v, f):
    id_unique = np.unique(f)
    v_unique = v[id_unique]
    id_old2new = np.ones(len(v)).astype(id_unique.dtype) * -1
    id_old2new[id_unique] = np.arange(len(v_unique))
    newf = id_old2new[f]
    return v_unique, newf


def save_obj(file, verts, faces, uvs, uvfaces, rm_unused=True):
    if rm_unused:
        verts, faces = cull_unused(verts, faces)
        uvs, uvfaces = cull_unused(uvs, uvfaces)

    with open(file, 'w') as f:
        for pos_color in verts:
            f.write(f"v {pos_color[0]} {pos_color[1]} {pos_color[2]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        faces1 = faces + 1
        uvfaces1 = uvfaces + 1
        for face, uvface in zip(faces1, uvfaces1):
            f.write(f"f {face[0]}/{uvface[0]} {face[1]}/{uvface[1]} {face[2]}/{uvface[2]}\n")

        f.write("\n")

