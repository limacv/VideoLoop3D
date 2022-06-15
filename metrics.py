from skimage import metrics
import torch
import torch.hub
from lpips.lpips import LPIPS
import os
import numpy as np

photometric = {
    "mse": None,
    "ssim": None,
    "psnr": None,
    "lpips": None
}

if os.name == 'posix':
    print("Change torch hub cache to /apdcephfs/private_leema/data/torch_cache")
    torch.hub.set_dir("/apdcephfs/private_leema/data/torch_cache")


def compute_img_metric(im1t: torch.Tensor, im2t: torch.Tensor,
                       metric="mse", mask=None, range01=True):
    """
    Args:
        im1t: tensor that has shape of batched images, *range from [-1, 1]*
        im2t: tensor that has shape of batched images, *range from [-1, 1]*
        metric: choose among mse, psnr, ssim, lpips
        mask: optional mask, tensor of shape [B, H, W] or [B, H, W, 1]
    """
    if metric not in photometric.keys():
        raise RuntimeError(f"img_utils:: metric {metric} not recognized")
    if photometric[metric] is None:
        if metric == "mse":
            photometric[metric] = metrics.mean_squared_error
        elif metric == "ssim":
            photometric[metric] = metrics.structural_similarity
        elif metric == "psnr":
            photometric[metric] = metrics.peak_signal_noise_ratio
        elif metric == "lpips":
            photometric[metric] = LPIPS().cpu()

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[1] == 1:
            mask = mask.permute(0, 2, 3, 1).cpu()
        batchsz, hei, wid, _ = mask.shape

    if range01:
        im1t = im1t * 2 - 1
        im2t = im2t * 2 - 1

    im1t = im1t.clamp(-1, 1).detach().cpu()
    im2t = im2t.clamp(-1, 1).detach().cpu()

    if im1t.shape[-1] == 3:
        im1t = im1t.permute(0, 3, 1, 2)
        im2t = im2t.permute(0, 3, 1, 2)

    if mask is not None:
        im1t = im1t * mask.permute(0, 3, 1, 2)
        im2t = im2t * mask.permute(0, 3, 1, 2)

    im1 = im1t.permute(0, 2, 3, 1).numpy()
    im2 = im2t.permute(0, 2, 3, 1).numpy()
    mask = mask.numpy()
    batchsz, hei, wid, _ = im1.shape
    values = []

    for i in range(batchsz):
        if metric in ["mse", "psnr"]:
            value = photometric[metric](
                im1[i], im2[i]
            )
            if mask is not None:
                pixelnum = mask[i, ..., 0].sum()
                if metric == "mse":
                    value = value * hei * wid / pixelnum
                else:
                    value = value - 10 * np.log10(hei * wid / pixelnum)
        elif metric in ["ssim"]:
            value, ssimmap = photometric["ssim"](
                im1[i], im2[i], multichannel=True, full=True
            )
            if mask is not None:
                value = (ssimmap * mask[i]).sum() / mask[i].sum() / 3
        elif metric in ["lpips"]:
            value = photometric[metric](
                im1t[i:i + 1], im2t[i:i + 1]
            )
        else:
            raise NotImplementedError
        values.append(value)

    return sum(values) / len(values)
