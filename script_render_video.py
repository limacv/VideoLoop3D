import os
import torch
import imageio
import numpy as np
import math
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from MPV import *

from dataloader import load_mv_videos, poses_avg
from utils import *
import shutil
from datetime import datetime
import cv2
from config_parser import config_parser
from tqdm import tqdm, trange
from copy import deepcopy


def evaluate(args):
    device = 'cuda:0'
    if args.gpu_num <= 0:
        device = 'cpu'
        print(f"Using CPU for training")

    expname = args.expname + args.expname_postfix
    print(f"Training: {expname}")
    datadir = os.path.join(args.prefix, args.datadir)
    expdir = os.path.join(args.prefix, args.expdir)
    videos, FPS, poses, intrins, bds, render_poses, render_intrins = \
        load_mv_videos(basedir=datadir,
                       factor=args.factor,
                           bd_factor=(args.near_factor, args.far_factor),
                       recenter=True)

    H, W = videos[0][0].shape[0:2]
    V = len(videos)
    print('Loaded llff', V, H, W, poses.shape, intrins.shape, render_poses.shape, bds.shape)

    ref_pose = poses_avg(poses)[:, :4]
    ref_extrin = pose2extrin_np(ref_pose)
    ref_intrin = intrins[0]
    ref_near, ref_far = bds.min(), bds.max()

    # Create nerf model
    if args.model_type == "MPMesh":
        nerf = MPMeshVid(args, H, W, ref_extrin, ref_intrin, ref_near, ref_far)
    else:
        raise RuntimeError(f"Unrecognized model type {args.model_type}")

    nerf = nn.DataParallel(nerf, list(range(args.gpu_num)))
    nerf.to(device)
    extrins = pose2extrin_np(poses)
    extrins = torch.tensor(extrins).float()
    poses = torch.tensor(poses).float()
    intrins = torch.tensor(intrins).float()

    ##########################
    # load from checkpoint
    ckpts = [os.path.join(expdir, expname, f)
             for f in sorted(os.listdir(os.path.join(expdir, expname))) if 'tar' in f]
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print(f"Using ckpt {ckpt_path}")
    else:
        raise RuntimeError("Failed, cannot find any ckpts")
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)

    state_dict = ckpt['network_state_dict']
    nerf.module.init_from_mpi(state_dict)
    nerf.to(device)

    # ##########################
    # start rendering
    # ##########################

    print('Begin')
    moviebase = os.path.join(expdir, expname, f'eval_')
    with torch.no_grad():
        nerf.eval()

        for viewi in range(V):
            torch.cuda.empty_cache()
            r_pose = extrins[viewi: viewi + 1]
            r_intrin = intrins[viewi: viewi + 1]
            rgb, extra = nerf(H, W, r_pose, r_intrin)
            rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()

            imageio.mimwrite(moviebase + f'_view{viewi:04d}.mp4', to8b(rgb), fps=25, quality=8)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    evaluate(args)

