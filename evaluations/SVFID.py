import numpy as np
import torch
from .C3D_model import C3D
from scipy import linalg


C3D_network = None


def calculate_batched_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[1])[None] * eps
    mats = (sigma1 + offset) @ (sigma2 + offset)
    # Product might be almost singular
    covmean = np.array([linalg.sqrtm(mat, disp=False)[0] for mat in mats])

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean, axis1=1, axis2=2)

    return ((diff * diff).sum(axis=-1) + np.trace(sigma1, axis1=1, axis2=2)
            + np.trace(sigma2, axis1=1, axis2=2) - 2 * tr_covmean)


def svfid(src, tar):
    """
    src/tar: tensor of F x H x W x 3, in (0, 255), rgb
    """
    global C3D_network
    if C3D_network is None:
        C3D_network = C3D()
        C3D_network.load_state_dict(torch.load('evaluations/c3d.pickle'))
        C3D_network.eval()

    C3D_network.to(src.device)
    with torch.no_grad():
        src = src.permute(3, 0, 1, 2)[None]
        tar = tar.permute(3, 0, 1, 2)[None]  # c, frm, h, w
        src_feat = C3D_network(src)
        tar_feat = C3D_network(tar)

        src_feat = src_feat[0, :50].permute(2, 3, 1, 0).flatten(0, 1)
        tar_feat = tar_feat[0, :50].permute(2, 3, 1, 0).flatten(0, 1)

        def batch_cov(points):
            B, N, D = points.size()
            mean = points.mean(dim=1, keepdims=True)
            diffs = (points - mean).reshape(B * N, D)
            prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
            bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
            return mean[:, 0], bcov  # (B, D, D)

        src_mean, src_cov = batch_cov(src_feat)
        tar_mean, tar_cov = batch_cov(tar_feat)

        fid = calculate_batched_frechet_distance(src_mean.cpu().numpy(),
                                                 src_cov.cpu().numpy(),
                                                 tar_mean.cpu().numpy(),
                                                 tar_cov.cpu().numpy())
        return fid.mean()
