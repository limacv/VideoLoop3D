from .lpips.lpips import LPIPS
import numpy as np

LPIPS_network = None


def _prepare_lpips(src, tar):
    global LPIPS_network
    if LPIPS_network is None:
        LPIPS_network = LPIPS()

    LPIPS_network.to(src.device)
    src = src.permute(0, 3, 1, 2) / (255 / 2) - 1
    tar = tar.permute(0, 3, 1, 2) / (255 / 2) - 1
    return src, tar


def compute_lpips(src, tar):
    """
    src/tar: tensor of F x H x W x 3, in (0, 255), rgb
    """
    global LPIPS_network
    src, tar = _prepare_lpips(src, tar)

    def compute_one_frame(frame, tar):
        scores = [LPIPS_network(frame, tar_[None]).item() for tar_ in tar]
        return min(scores)

    err = [compute_one_frame(f[None], tar) for f in src]
    return np.array(err).mean()


def compute_lpips_slidewindow(src, tar):
    """
    src/tar: tensor of F x H x W x 3, in (0, 255), rgb
    """
    global LPIPS_network
    if len(src) > len(tar):
        src, tar = tar, src
    src, tar = _prepare_lpips(src, tar)

    def compute_aligned_lpips(s, t):
        scores = [LPIPS_network(sf[None], tf[None]).item() for sf, tf in zip(s, t)]
        return np.mean(scores)

    err = [compute_aligned_lpips(src, tar[i: i + len(src)]) for i in range(len(tar) - len(src))]
    return np.array(err).min()
