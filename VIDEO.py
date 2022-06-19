import torch
import torch.nn as nn
import torch.nn.functional as torchf
import os
import imageio
import time
import cv2
from utils import *
from utils_mpi import *
import trimesh


activate = {'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'none': lambda x: x,
            'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
            'softplus': lambda x: nn.Softplus()(x - 1),
            'tanh': torch.tanh,
            'clamp': lambda x: torch.clamp(x, 0, 1)}


class SimpleVideo(nn.Module):
    def __init__(self, args, H, W, time_len, ref_intrin):
        super(SimpleVideo, self).__init__()
        self.args = args
        self.h, self.w = H, W
        self.time_len = time_len
        video = torch.rand((time_len, 3, self.mpi_h, self.mpi_w))
        self.register_parameter("video", nn.Parameter(video, requires_grad=True))

    def forward(self, h, w, tar_extrins, tar_intrins):
        with torch.no_grad():
            homo = compute_homography(ref_extrins, ref_intrins, tar_extrins, tar_intrins,
                                      self.plane_normal, self.plane_depth)
        mpi_warp = warp_homography(h, w, homo, self.tonemapping(self.mpi))

        extra = {}
        if self.training:
            if self.args.sparsity_loss_weight > 0:
                sparsity = mpi_warp[:, :, -1].mean()
                extra["sparsity"] = sparsity.reshape(1, -1)
        return overcomposeNto0(mpi_warp), extra

