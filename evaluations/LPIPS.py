from .lpips.lpips import LPIPS
import numpy as np

LPIPS_network = None


def compute_lpips(src, tar):
    """
    src/tar: tensor of F x H x W x 3, in (0, 255), rgb
    """
    global LPIPS_network
    if LPIPS_network is None:
        LPIPS_network = LPIPS()

    LPIPS_network.to(src.device)
    src = src.permute(0, 3, 1, 2) / (255 / 2) - 1
    tar = tar.permute(0, 3, 1, 2) / (255 / 2) - 1

    def compute_one_frame(frame, tar):
        scores = [LPIPS_network(frame, tar_[None]).item() for tar_ in tar]
        return min(scores)

    err = [compute_one_frame(f[None], tar) for f in src]
    return np.array(err).mean()
