import torch
import torch.nn.functional as F
# from unfoldNd import UnfoldNd
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


class Patch3DSWDLoss(torch.nn.Module):
    def __init__(self, patch_size=7, patcht_size=7, stride=1, dilate=1, num_proj=256, use_convs=True, mask_patches_factor=0,
                 roi_region_pct=0.02):
        super(Patch3DSWDLoss, self).__init__()
        self.name = f"Conv3DSWDLoss(p-{patch_size}:{stride})"
        self.patch_size = patch_size
        self.patcht_size = patcht_size
        self.stride = stride
        self.dilate = dilate
        self.num_proj = num_proj
        self.use_convs = use_convs
        self.mask_patches_factor = mask_patches_factor
        self.roi_region_pct = roi_region_pct

    def forward(self, x, y, mask=None):
        b, c, f, h, w = x.shape

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
            projx = F.conv3d(x, rand, dilation=self.dilate)
            _, _, cf, ch, cw = projx.shape
            projx = projx.reshape(self.num_proj, cf, ch * cw)
            projy = F.conv3d(y, rand, dilation=self.dilate).reshape(self.num_proj, -1, ch * cw)
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

