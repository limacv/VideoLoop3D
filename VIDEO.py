import torch
import torch.nn as nn
import torch.nn.functional as torchf
import os
import imageio
import time
import cv2
from utils_vid import *


activate = {'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'none': lambda x: x,
            'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
            'softplus': lambda x: nn.Softplus()(x - 1),
            'tanh': torch.tanh,
            'clamp': lambda x: torch.clamp(x, 0, 1),
            'clamp_g': lambda x: x + (torch.clamp(x, 0, 1) - x).detach(),
            }


class SimpleTargetVideo(nn.Module):
    def __init__(self):
        super(SimpleTargetVideo, self).__init__()
        pass  # TODO: to be implemented


class SimpleVideo(nn.Module):
    def __init__(self, args, H, W, T):
        super(SimpleVideo, self).__init__()
        self.args = args
        self.h, self.w = H, W
        self.t = T
        video = torch.rand((1, 3, T, self.h, self.w))
        self.register_parameter("video", nn.Parameter(video, requires_grad=True))
        self.isloop = True

        # loss related
        self.swd_patch_size = args.swd_patch_size
        self.swd_patcht_size = args.swd_patcht_size
        self.swd_loss = Patch3DSWDLoss(
            patch_size=self.swd_patch_size,
            patcht_size=self.swd_patcht_size, stride=1,
            num_proj=256, use_convs=True, mask_patches_factor=0,
            roi_region_pct=0.02
        )
        self.activate = activate[args.rgb_activate]

    @torch.no_grad()
    def export_video(self, prefix):
        video = self.video[0].permute(1, 2, 3, 0)
        video = (video * 255).clamp(0, 255).type(torch.uint8).cpu().numpy()
        imageio.mimwrite(f"{prefix}.mp4", video, fps=25)

    def forward(self, h, w, h_start: torch.Tensor=0, w_start=0, res=None):
        if isinstance(h_start, int):
            h_start, w_start = [h_start], [w_start]
        elif isinstance(h_start, torch.Tensor):
            h_start, w_start = h_start.tolist(), w_start.tolist()

        patches = []
        for h_s, w_s in zip(h_start, w_start):
            # update roi
            h_crop = min(self.h, (h_s + h)) - h_s
            w_crop = min(self.w, (w_s + w)) - w_s
            h_pad = h - h_crop
            w_pad = w - w_crop
            patch = self.video[..., h_s:h_s + h_crop, w_s:w_s + w_crop]
            patch = torchf.pad(patch, [0, w_pad, 0, h_pad])
            patches.append(patch)
        patches = torch.cat(patches)  # B, C, Frame, H, W

        if self.isloop:
            pad_frame = self.swd_patcht_size - 1
            patches = torch.cat([patches, patches[:, :, :pad_frame]], 2)

        video = self.activate(patches)
        if self.training:
            extra = {}
            assert res is not None

            # main loss
            swd_loss = self.swd_loss(video * 2 - 1, res * 2 - 1)
            extra['swd'] = swd_loss.reshape(-1, 1)

            return None, extra

        else:
            return video, {}

