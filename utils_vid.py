import torch
import torch.nn.functional as F
from unfoldNd import UnfoldNd, FoldNd
import numpy as np


def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates entries of the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2


def extract_patches(x, patch_size, stride):
    """Extract normalized patches from an image"""
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 3, patch_size, patch_size)
    return x_patches.view(b, -1, 3 * patch_size ** 2)


def extract_3Dpatches(x, patch_size, tpatch_size, stride, tstride):
    """Extract normalized patches from an image"""
    b, c, d, h, w = x.shape
    d_out = (d - (tpatch_size - 1) - 1) // tstride + 1
    h_out = (h - (patch_size - 1) - 1) // stride + 1
    w_out = (w - (patch_size - 1) - 1) // stride + 1
    unfold = UnfoldNd(kernel_size=(patch_size, patch_size, tpatch_size),
                      stride=(stride, stride, tstride))
    x_patches = unfold(x).reshape(b, -1, d_out, h_out, w_out)
    return x_patches


def efficient_compute_distances(X, Y):
    """
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (b, n1, d) tensor
    :param Y:  (b, n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(-1)[:, :, None] + (Y * Y).sum(-1)[:, None, :] - 2.0 * (X @ Y.permute(0, 2, 1))
    d = X.shape[-1]
    dist /= d  # normalize by size of vector to make dists independent of the size of d
               # ( use same alpha for all patche-sizes)
    return dist


def get_col_mins_efficient(X, Y, chunksz):
    """
    Computes the l2 distance to the closest x or each y.
    :param X:  (B, n1, d) tensor
    :param Y:  (B, n2, d) tensor
    Returns (B, n2)
    """
    mins = torch.zeros(Y.shape[:2], dtype=X.dtype, device=X.device)
    for starti in range(0, len(Y), chunksz):
        mins[:, starti: starti + chunksz] = efficient_compute_distances(X, Y[:, starti: starti + chunksz]).min(1)[0]
    return mins


def get_NN_indices_low_memory(X, Y, alpha, chunksz):
    """
    Get the nearest neighbor index from Y for each X.
    Avoids holding a (n1 * n2) amtrix in order to reducing memory footprint to (b * max(n1,n2)).
    :param X:  (B, n1, d) tensor
    :param Y:  (B, n2, d) tensor
    Returns (B, n1), long tensor of indice to Y
    """
    if alpha is not None:
        normalizing_row = get_col_mins_efficient(X, Y, chunksz=chunksz)
        normalizing_row = alpha + normalizing_row[:, None]
    else:
        normalizing_row = 1

    NNs = torch.zeros(X.shape[:2], dtype=torch.long, device=X.device)
    for starti in range(0, X.shape[1], chunksz):
        dists = efficient_compute_distances(X[:, starti: starti + chunksz], Y)
        dists = dists / normalizing_row
        NNs[:, starti: starti + chunksz] = torch.argmin(dists, dim=2)
    return NNs


class Patch3DSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, patcht_size=7, stride=1, stridet=1, num_proj=256, use_convs=True, mask_patches_factor=0,
                       roi_region_pct=0.02):
        super(Patch3DSWDLoss, self).__init__()
        self.name = f"Conv3DSWDLoss(p-{patch_size}:{stride})"
        self.patch_size = patch_size
        self.patcht_size = patcht_size
        self.stride = stride
        self.stridet = stridet
        self.num_proj = num_proj
        self.use_convs = True
        self.mask_patches_factor = mask_patches_factor
        self.roi_region_pct = roi_region_pct

    def forward(self, x, y, mask=None, same_input=False, alpha=None):
        b, c, f, h, w = x.shape
        x = x * 2 - 1
        y = y * 2 - 1
        # crop version
        # crop_size = int(np.sqrt(h * w * self.roi_region_pct)) + self.patch_size - 1
        # crop_x1 = np.random.randint(0, w - crop_size + 1)
        # crop_y1 = np.random.randint(0, h - crop_size + 1)
        # x = x[..., crop_y1:crop_y1 + crop_size, crop_x1:crop_x1 + crop_size]
        # y = y[..., crop_y1:crop_y1 + crop_size, crop_x1:crop_x1 + crop_size]

        assert b == 1, "Batches not implemented"
        rand = torch.randn(self.num_proj, 3, self.patcht_size, self.patch_size, self.patch_size).to(
            x.device)  # (slice_size**2*ch)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

        if self.use_convs:
            projx = F.conv3d(x, rand,
                             stride=[self.stride, self.stride, self.stridet])
            _, _, cf, ch, cw = projx.shape
            projx = projx.reshape(self.num_proj, cf, ch*cw)
            projy = F.conv3d(y, rand,
                             stride=[self.stride, self.stride, self.stridet]).reshape(self.num_proj, -1, ch*cw)
            projx = projx.permute(0, 2, 1).reshape(self.num_proj * ch * cw, -1)
            projy = projy.permute(0, 2, 1).reshape(self.num_proj * ch * cw, -1)
        else:
            projx = torch.matmul(extract_patches(x, self.patch_size, self.stride), rand)
            projy = torch.matmul(extract_patches(y, self.patch_size, self.stride), rand)

        if mask is not None:
            # duplicate patches that touches the mask by a factor

            mask_patches = extract_patches(mask, self.patch_size, self.stride)[0]  # in [-1,1]
            mask_patches = torch.any(mask_patches > 0, dim=-1)
            projy = torch.cat([projy[:, ~mask_patches]] + [projy[:, mask_patches]] * self.mask_patches_factor, dim=1)

        projx, projy = duplicate_to_match_lengths(projx, projy)

        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()

        return loss


class Patch3DGPNNDirectLoss(torch.nn.Module):
    def __init__(self, patch_size=7, patcht_size=7, stride=1, stridet=1):
        super().__init__()
        self.name = f"Patch3DGPNN"
        self.patch_size = patch_size
        self.patcht_size = patcht_size
        self.stride = stride
        self.stridet = stridet
        self.last_y2x = None
        self.last_weight = None

    def forward(self, x, y, mask=None, same_input=False, alpha=1e10):  # x is the src and y is the target
        if same_input:
            weight = self.last_weight
            y2x = self.last_y2x
        else:
            alpha = None if alpha > 1000 else alpha

            with torch.no_grad():
                projx = extract_3Dpatches(x, self.patch_size, self.patcht_size, self.stride, self.stridet)  # b, c, d, h, w
                b, c, d, h, w = projx.shape
                B = b * h * w
                D = d * h * w
                projx = projx.permute(0, 3, 4, 2, 1).reshape(B, -1, c)
                projy = extract_3Dpatches(y, self.patch_size, self.patcht_size, self.stride, self.stridet)  # b, c, d, h, w
                projy = projy.permute(0, 3, 4, 2, 1).reshape(B, -1, c)
                nns = get_NN_indices_low_memory(projx, projy, alpha, 1024)
                projy2x = projy[torch.arange(B, device=nns.device)[:, None], nns]
                fold = FoldNd(x.shape[-3:],
                              kernel_size=(self.patch_size, self.patch_size, self.patcht_size),
                              stride=(self.stride, self.stride, self.stridet))
                projy2x_unfold = projy2x.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)
                projy2x_unfold = projy2x_unfold.reshape(b,
                                                        3, self.patcht_size, self.patch_size, self.patch_size,
                                                        D)
                weight = torch.ones_like(projy2x_unfold[:, :1])
                projy2x_unfold = torch.cat([projy2x_unfold, weight], dim=1)
                y2x = fold(projy2x_unfold.reshape(b, -1, D))
                weight = y2x[:, 3:].clamp_min(1e-10)
                y2x = y2x[:, :3] / weight
                self.last_weight = weight
                self.last_y2x = y2x

        loss = (torch.abs(x - y2x) * weight).mean()
        return loss


class Patch3DMSE(torch.nn.Module):  # dummy loss for ablation
    def __init__(self):
        super().__init__()
        self.name = f"MSE"

    def forward(self, x, y, mask=None, same_input=False, alpha=None):  # x is the src and y is the target
        loss = torch.abs(x - y).mean()
        return loss
