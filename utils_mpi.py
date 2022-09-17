import torch
import torch.nn.functional as torchf
import numpy as np
from typing import Union, Sequence, Tuple
import torch.nn as nn


class Feat2RGBMLP_alpha(nn.Module):  # alpha is view-independent
    def __init__(self, feat_cnl, view_cn):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_cnl + view_cn - 1, 48), nn.ReLU(),
            nn.Linear(48, 3)
        )

    def forward(self, x):
        return torch.cat([self.mlp(x[..., 1:]), x[..., :1]], dim=-1)


class NeX_RGBA(nn.Module):  # alpha is view-independent
    def __init__(self, feat_cnl, view_cn):
        assert feat_cnl % 4 == 0
        super().__init__()
        self.feat_cnl, self.view_cnl = feat_cnl, view_cn
        self.mlp = nn.Sequential(
            nn.Linear(view_cn, 64), nn.ReLU(),
            nn.Linear(64, feat_cnl - 4)
        )

    def forward(self, x):
        basis = self.mlp(x[:, self.feat_cnl:]).reshape(-1, self.feat_cnl // 4 - 1, 4)
        return (basis * x[:, 4:self.feat_cnl].reshape(-1, self)).sum(dim=-2) + x[:, :4]


class NeX_RGB(nn.Module):  # alpha is view dependent
    def __init__(self, feat_cnl, view_cn):
        super().__init__()
        self.feat_cnl, self.view_cnl = feat_cnl, view_cn
        self.mlp = nn.Sequential(
            nn.Linear(view_cn, 64), nn.ReLU(),
            nn.Linear(64, 3 * (feat_cnl - 1))
        )

    def forward(self, x):
        basis = self.mlp(x[:, self.feat_cnl:]).reshape(-1, self.feat_cnl - 1, 4)
        rgb = (basis * x[:, 1:self.feat_cnl, None]).sum(dim=-2)
        return torch.cat([rgb, x[..., :1]], dim=-1)


class SphericalHarmoic_RGB(nn.Module):  # alpha is view-independent
    def __init__(self, feat_cnl, view_cn):
        super().__init__()
        self.sh_dim = feat_cnl // 3
        self.feat_cnl = feat_cnl
        self.view_cnl = view_cn

    def forward(self, x):
        feat, view = torch.split(x, [self.feat_cnl, self.view_cnl], -1)
        sh_base = eval_sh_bases(self.sh_dim, view[..., :3])
        rgb = torch.sum(sh_base.reshape(-1, 1, self.sh_dim) * feat[..., 1:].reshape(-1, 3, self.sh_dim), dim=-1)
        return torch.cat([rgb, feat[..., :1]], dim=-1)


class SphericalHarmoic_RGBA(nn.Module):  # alpha is view-independent
    def __init__(self, feat_cnl, view_cn):
        super().__init__()
        self.sh_dim = 9
        self.feat_cnl = feat_cnl
        self.view_cnl = view_cn

    def forward(self, x):
        feat, view = torch.split(x, [self.feat_cnl, self.view_cnl], -1)
        sh_base = eval_sh_bases(self.sh_dim, view[..., :3])
        rgba = torch.sum(sh_base.reshape(1, 1, -1) * feat.reshape(-1, 4, self.sh_dim), dim=-1)
        return rgba


# geometric utils: generating geometry
# #####################################
def gen_mpi_vertices(H, W, intrin, num_vert_h, num_vert_w, planedepth):
    verts = torch.meshgrid(
        [torch.linspace(0, H - 1, num_vert_h), torch.linspace(0, W - 1, num_vert_w)])
    verts = torch.stack(verts[::-1], dim=-1).reshape(1, -1, 2)
    # num_plane, H*W, 2
    verts = (verts - intrin[None, None, :2, 2]) * planedepth[:, None, None].type_as(verts)
    verts /= intrin[None, None, [0, 1], [0, 1]]
    zs = planedepth[:, None, None].expand_as(verts[..., :1])
    verts = torch.cat([verts.reshape(-1, 2), zs.reshape(-1, 1)], dim=-1)
    return verts


def overcompose(alpha, content):
    """
    compose mpi back (-1) to front (0)
    alpha: [B, H, W, 32]
    content: [B, H, W, 32, C]
    """
    batchsz, num_plane, height, width, _ = content.shape

    blendweight = torch.cumprod((- alpha + 1)[..., :-1], dim=-1)  # B x H x W x LayerNum-1
    blendweight = torch.cat([
        alpha[..., :1],
        alpha[..., 1:] * blendweight
        ], dim=-1)

    rgb = (content * blendweight.unsqueeze(-1)).sum(dim=-2)
    return rgb, blendweight


def overcomposeNto0(mpi: torch.Tensor, blendweight=None, ret_mask=False, blend_content=None) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    compose mpi back to front
    mpi: [B, 32, 4, H, W]
    blendweight: [B, 32, H, W] for reduce reduntant computation
    blendContent: [B, layernum, cnl, H, W], if not None,
    return: image of shape [B, 4, H, W]
        [optional: ] mask of shape [B, H, W], soft mask that
    """
    batchsz, num_plane, _, height, width = mpi.shape
    alpha = mpi[:, :, -1, ...]  # alpha.shape == B x LayerNum x H x W
    if blendweight is None:
        blendweight = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                 torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    renderw = alpha * blendweight

    content = mpi[:, :, :3, ...] if blend_content is None else blend_content
    rgb = (content * renderw.unsqueeze(2)).sum(dim=1)
    if ret_mask:
        return rgb, blendweight
    else:
        return rgb


def estimate_disparity_np(mpi: np.ndarray, min_depth=1, max_depth=100):
    """Compute disparity map from a set of MPI layers.
    mpi: np.ndarray or torch.Tensor
    From reference view.

    Args:
      layers: [..., L, H, W, C+1] MPI layers, back to front.
      depths: [..., L] depths for each layer.

    Returns:
      [..., H, W, 1] Single-channel disparity map from reference viewpoint.
    """
    num_plane, height, width, chnl = mpi.shape
    disparities = np.linspace(1. / max_depth, 1. / min_depth, num_plane)
    disparities = disparities.reshape(-1, 1, 1, 1)

    alpha = mpi[..., -1:]
    alpha = alpha * np.concatenate([np.cumprod(1 - alpha[::-1], axis=0)[::-1][1:],
                                    np.ones([1, height, width, 1])], axis=0)
    disparity = (alpha * disparities).sum(axis=0)
    # Weighted sum of per-layer disparities:
    return disparity.squeeze(-1)


def warp_homography(h, w, homos: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    """
    apply differentiable homography
    h, w: the target size
    homos: [B x D x 3 x 3]
    images: [B x D x 4 x H x W]
    """
    batchsz, planenum, cnl, hei, wid = images.shape
    y, x = torch.meshgrid([torch.arange(h), torch.arange(w)])
    x, y = x.type_as(images), y.type_as(images)
    one = torch.ones_like(x)
    grid = torch.stack([x, y, one], dim=-1)  # grid: B x D x H x W x 3
    new_grid = homos.unsqueeze(-3).unsqueeze(-3) @ grid.unsqueeze(-1)
    new_grid = (new_grid.squeeze(-1) / new_grid[..., 2:3, 0])[..., 0:2]  # grid: B x D x H x W x 2
    new_grid = new_grid / torch.tensor([wid / 2, hei / 2]).type_as(new_grid) - 1.
    warpped = torchf.grid_sample(images.reshape(batchsz * planenum, cnl, hei, wid),
                                 new_grid.reshape(batchsz * planenum, h, w, 2), align_corners=True)
    return warpped.reshape(batchsz, planenum, cnl, h, w)


def warp_homography_withdepth(homos: torch.Tensor, images: torch.Tensor, depth: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Please note that homographies here are not scale invariant. make sure that rotation matrix has 1 det. R.det() == 1.
    apply differentiable homography
    homos: [B x D x 3 x 3]
    images: [B x D x 4 x H x W]
    depth: [B x D] or [B x D x 1] (depth in *ref space*)
    ret:
        the warpped mpi
        the warpped depth
    """
    batchsz, planenum, cnl, hei, wid = images.shape
    y, x = torch.meshgrid([torch.arange(hei), torch.arange(wid)])
    x, y = x.type_as(images), y.type_as(images)
    one = torch.ones_like(x)
    grid = torch.stack([x, y, one], dim=-1).reshape(1, 1, hei, wid, 3, 1)
    if depth.dim() == 4:
        depth = depth.reshape(batchsz, planenum, 1, hei, wid)
    else:
        depth = depth.reshape(batchsz, planenum, 1, 1, 1)

    new_grid = homos.unsqueeze(-3).unsqueeze(-3) @ grid
    new_depth = depth.reshape(batchsz, planenum, 1, 1) / new_grid[..., -1, 0]
    new_grid = (new_grid.squeeze(-1) / new_grid[..., 2:3, 0])[..., 0:2]  # grid: B x D x H x W x 2
    new_grid = new_grid / torch.tensor([wid / 2, hei / 2]).type_as(new_grid) - 1.
    warpped = torchf.grid_sample(images.reshape(batchsz * planenum, cnl, hei, wid),
                                 new_grid.reshape(batchsz * planenum, hei, wid, 2), align_corners=True)
    return warpped.reshape(batchsz, planenum, cnl, hei, wid), new_depth


def make_depths(num_plane, min_depth, max_depth):
    return torch.reciprocal(torch.linspace(1. / max_depth, 1. / min_depth, num_plane, dtype=torch.float32))


def estimate_disparity_torch(mpi: torch.Tensor, depthes: torch.Tensor, blendweight=None, retbw=False):
    """Compute disparity map from a set of MPI layers.
    mpi: tensor of shape B x LayerNum x 4 x H x W
    depthes: tensor of shape [B x LayerNum] or [B x LayerNum x H x W] (means different depth for each pixel]
    blendweight: optional blendweight that to reduce reduntante computation
    return: tensor of shape B x H x W
    """
    assert (mpi.dim() == 5)
    batchsz, num_plane, _, height, width = mpi.shape
    disparities = torch.reciprocal(depthes)
    if disparities.dim() != 4:
        disparities = disparities.reshape(batchsz, num_plane, 1, 1).type_as(mpi)

    alpha = mpi[:, :, -1, ...]  # alpha.shape == B x LayerNum x H x W
    if blendweight is None:
        blendweight = torch.cat([torch.cumprod(- torch.flip(alpha, dims=[1]) + 1, dim=1).flip(dims=[1])[:, 1:],
                                 torch.ones([batchsz, 1, height, width]).type_as(alpha)], dim=1)
    renderweight = alpha * blendweight
    disparity = (renderweight * disparities).sum(dim=1)

    if retbw:
        return disparity, blendweight
    else:
        return disparity


def compute_homography(src_extrin_4x4: torch.Tensor, src_intrin: torch.Tensor,
                       tar_extrin_4x4: torch.Tensor, tar_intrin: torch.Tensor,
                       normal: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """
    compute homography matrix, detail pls see https://en.wikipedia.org/wiki/Homography_(computer_vision)
        explanation: assume P, P1, P2 be coordinate of point in plane in world, ref, tar space
        P1 = R1 @ P + t1               P2 = R2 @ P + t2
            so P1 = R @ P2 + t   where:
                R = R1 @ R2^T, t = t1 - R @ t2
        n^T @ P1 = d be plane equation in ref space,
            so in tar space: n'^T @ P2 = d'  where:
                n' = R^T @ n,    d' = d - n^T @ t

        so P1 = R @ P2 + d'^-1 t @ n'T @ P2 = (R + t @ n'^T @ R / (d - n^T @ t)) @ P2
    src_extrin/tar_extrin: [B, 3, 4] = [R | t]
    src_intrin/tar_intrin: [B, 3, 3]
    normal: [B, D, 3] normal of plane in *reference space*
    distances: [B, D] offset of plaen in *ref space*
        so the plane equation: n^T @ P1 = d  ==>  n'^T
    return: [B, D, 3, 3]
    """
    batchsz, _, _ = src_extrin_4x4.shape
    pose = src_extrin_4x4 @ torch.inverse(tar_extrin_4x4)
    # rotation = R1 @ R2^T
    # translation = (t1 - R1 @ R2^T @ t2)
    rotation, translation = pose[..., :3, :3], pose[..., :3, 3:].squeeze(-1)
    distances_tar = -(normal @ translation.unsqueeze(-1)).squeeze(-1) + distances

    # [..., 3, 3] -> [..., D, 3, 3]
    # multiply extra rotation because normal is in reference space
    homo = rotation.unsqueeze(-3) + (translation.unsqueeze(-1) @ normal.unsqueeze(-2) @ rotation.unsqueeze(-3)) \
           / distances_tar.unsqueeze(-1).unsqueeze(-1)
    homo = src_intrin.unsqueeze(-3) @ homo @ torch.inverse(tar_intrin.unsqueeze(-3))
    return homo


def render_newview(mpi: torch.Tensor, srcextrin: torch.Tensor, tarextrin: torch.Tensor,
                   srcintrin: torch.Tensor, tarintrin: torch.Tensor,
                   depths: torch.Tensor, ret_mask=False, ret_dispmap=False) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
                 Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """
    mpi: [B, 32, 4, H, W]
    srcpose&tarpose: [B, 3, 4]
    depthes: tensor of shape [Bx LayerNum]
    intrin: [B, 3, 3]
    ret: ref_view_images[, mask][, disparitys]
    """
    batchsz, planenum, _, hei, wid = mpi.shape

    planenormal = torch.tensor([0, 0, 1]).reshape(1, 3).repeat(batchsz, 1).type_as(mpi)
    distance = depths.reshape(batchsz, planenum)
    with torch.no_grad():
        # switching the tar/src pose since we have extrinsic but compute_homography uses poses
        # srcextrin = torch.tensor([1, 0, 0, 0,  # for debug usage
        #                           0, 1, 0, 0,
        #                           0, 0, 1, 0]).reshape(1, 3, 4).type_as(intrin)
        # tarextrin = torch.tensor([np.cos(0.3), -np.sin(0.3), 0, 0,
        #                           np.sin(0.3), np.cos(0.3), 0, 0,
        #                           0, 0, 1, 1.5]).reshape(1, 3, 4).type_as(intrin)
        homos = compute_homography(srcextrin, srcintrin, tarextrin, tarintrin,
                                   planenormal, distance)
    if not ret_dispmap:
        mpi_warp = warp_homography(homos, mpi)
        return overcomposeNto0(mpi_warp, ret_mask=ret_mask)
    else:
        mpi_warp, mpi_depth = warp_homography_withdepth(homos, mpi, distance)
        disparitys = estimate_disparity_torch(mpi_warp, mpi_depth)
        return overcomposeNto0(mpi_warp, ret_mask=ret_mask), disparitys


def warp_flow(content: torch.Tensor, flow: torch.Tensor, offset=None, pad_mode="zeros", mode="bilinear"):
    """
    content: [..., cnl, H, W]
    flow: [..., 2, H, W]
    """
    assert content.dim() == flow.dim()
    orishape = content.shape
    cnl, hei, wid = content.shape[-3:]
    mpi = content.reshape(-1, cnl, hei, wid)
    flow = flow.reshape(-1, 2, hei, wid).permute(0, 2, 3, 1)

    if offset is None:
        y, x = torch.meshgrid([torch.arange(hei), torch.arange(wid)])
        x, y = x.type_as(mpi), y.type_as(mpi)
        offset = torch.stack([x, y], dim=-1)
    grid = offset.reshape(1, hei, wid, 2) + flow
    normanator = torch.tensor([(wid - 1) / 2, (hei - 1) / 2]).reshape(1, 1, 1, 2).type_as(grid)
    warpped = torchf.grid_sample(mpi, grid / normanator - 1., padding_mode=pad_mode, mode=mode, align_corners=True)
    return warpped.reshape(orishape)


# spherical hamoric related, copy from svox2

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y
        result[..., 2] = SH_C1 * z
        result[..., 3] = -SH_C1 * x
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy
            result[..., 5] = SH_C2[1] * yz
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy)
            result[..., 7] = SH_C2[3] * xz
            result[..., 8] = SH_C2[4] * (xx - yy)

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy)
                result[..., 10] = SH_C3[1] * xy * z
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy)
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy)
                result[..., 14] = SH_C3[5] * z * (xx - yy)
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy)

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy)
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy)
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1)
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3)
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3)
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3)
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1)
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy)
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return result
