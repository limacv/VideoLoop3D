import torch
import torch.nn as nn
import torch.nn.functional as torchf
import os
import imageio
import time
import cv2
from utils import *
from utils_mpi import *

activate = {'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'none': lambda x: x,
            'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
            'softplus': lambda x: nn.Softplus()(x - 1),
            'tanh': torch.tanh}


class MPI(nn.Module):
    def __init__(self, args, H, W, ref_extrin, ref_intrin, near, far):
        super(MPI, self).__init__()
        self.args = args
        self.mpi_h, self.mpi_w = int(args.mpi_h_scale * H), int(args.mpi_w_scale * W)
        self.mpi_d, self.near, self.far = args.mpi_d, near, far
        self.H, self.W = H, W
        self.H_start, self.W_start = (self.mpi_h - H) // 2, (self.mpi_w - W) // 2
        assert ref_extrin.shape == (4, 4) and ref_intrin.shape == (3, 3)
        ref_intrin_mpi = get_new_intrin(ref_intrin, - self.H_start, - self.W_start)
        self.register_buffer("ref_extrin", torch.tensor(ref_extrin))
        self.register_buffer("ref_intrin", torch.tensor(ref_intrin_mpi).float())

        planenormal = torch.tensor([0, 0, 1]).reshape(1, 3).float()
        self.register_buffer("plane_normal", planenormal)

        planedepth = make_depths(self.mpi_d, near, far).float()
        self.register_buffer("plane_depth", planedepth)

        mpi = torch.rand((1, self.mpi_d, 4, self.mpi_h, self.mpi_w))  # RGBA
        mpi[:, :, -1] = -2
        # TODO: initalization of the MPI
        self.register_parameter("mpi", nn.Parameter(mpi, requires_grad=True))
        self.tonemapping = activate['sigmoid']

    def forward(self, h, w, tar_extrins, tar_intrins):
        ref_extrins = self.ref_extrin[None, ...].expand_as(tar_extrins)
        ref_intrins = self.ref_intrin[None, ...].expand_as(tar_intrins)
        with torch.no_grad():
            homo = compute_homography(ref_extrins, ref_intrins, tar_extrins, tar_intrins,
                                      self.plane_normal, self.plane_depth)
        mpi_warp = warp_homography(h, w, homo, self.tonemapping(self.mpi))

        extra = {}
        if self.training:
            if self.args.sparsity_loss_weight > 0:
                sparsity = mpi_warp[:, :, -1].mean()
                extra["sparsity"] = sparsity.reshape(1, -1)
        return overcompose(mpi_warp), extra
