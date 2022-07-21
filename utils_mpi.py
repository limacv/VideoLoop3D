import torch
import torch.nn.functional as torchf
import numpy as np
from typing import Union, Sequence, Tuple


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
