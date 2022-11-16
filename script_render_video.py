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
    print(f"Rendering: {expname}")
    datadir = os.path.join(args.prefix, args.datadir)
    expdir = os.path.join(args.prefix, args.expdir)

    # figure out render_frm to be consistent
    render_frm = args.f if args.f > 0 else (120 // args.mpv_frm_num + 1) * args.mpv_frm_num
    print(f"loading render pose with {render_frm} frames")
    videos, FPS, poses, intrins, bds, render_poses, render_intrins = \
        load_mv_videos(basedir=datadir,
                       factor=args.factor,
                       bd_factor=(args.near_factor, args.far_factor),
                       recenter=True,
                       render_frm=render_frm,
                       render_scaling=args.render_scaling)

    H, W = videos[0][0].shape[0:2]
    V = len(videos)
    print('Loaded llff', V, H, W, poses.shape, intrins.shape, render_poses.shape, bds.shape)

    # figure out view to be rendered
    view_poses, view_intrins = render_poses.copy(), render_intrins.copy()
    render_t = np.arange(len(render_poses)) % args.mpv_frm_num
    if args.v == 'test':
        args.v = args.test_view_idx.split(',')[0]

    if len(args.v) > 0:
        render_t = render_t[:args.mpv_frm_num]
        if args.v[0] == 'r':
            v = int(args.v[1:])
            view_poses[:] = view_poses[v:v+1]
            view_intrins[:] = render_intrins[v:v+1]
            print(f"Rendering view {v} in render_pose")
        else:
            v = int(args.v)
            view_poses[:] = poses[v:v+1]
            view_intrins[:] = intrins[v:v+1]
            print(f"Rendering view {v}")

    # figure out time to be rendered
    if len(args.t) > 0:
        if ',' in args.t and ':' not in args.t:
            time_range = list(map(int, args.t.split(',')))
            render_t = render_t[time_range]
        elif ':' in args.t:
            slices = args.t.split(',')
            render_t = []
            for slic in slices:
                start, end = list(map(int, slic.split(':')))
                step = 1 if start <= end else -1
                render_t.append(np.arange(start, end, step))
            render_t = np.concatenate(render_t)
        else:
            time_range = [int(args.t)]
            render_t = render_t[time_range]

    view_poses = view_poses[:len(render_t)]
    view_intrins = view_intrins[:len(render_t)]
    print(f"Rendering time {render_t}")

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

    view_extrins = pose2extrin_np(view_poses)
    view_extrins = torch.tensor(view_extrins).float()
    view_intrins = torch.tensor(view_intrins).float()

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
    moviebase = os.path.join(expdir, expname, f'renderonly')
    os.makedirs(moviebase, exist_ok=True)
    with torch.no_grad():
        nerf.eval()

        rgbs = []
        for viewi in trange(len(view_poses)):
            r_pose = view_extrins[viewi: viewi + 1]
            r_intrin = view_intrins[viewi: viewi + 1]
            t = render_t[viewi: viewi + 1]
            rgb, extra = nerf(H, W, r_pose, r_intrin, t)
            rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()[0]
            rgbs.append(to8b(rgb))

        if len(rgbs) < 3:
            print("too less frames, force to write images")
            args.type += 'seq'

        if 'seq' in args.type:
            for i, rgb in enumerate(rgbs):
                imageio.imwrite(moviebase + f'/view{args.v}t{args.t}_{i:04d}.png', rgb)
        else:
            imageio.mimwrite(moviebase + f'/view{args.v}t{args.t}.mp4', rgbs, fps=25, quality=8, macro_block_size=1)


if __name__ == '__main__':
    parser = config_parser()
    parser.add_argument("--v", type=str, default='',
                        help='render view control, empty to be render_pose, r# to be #-th render pose, '
                             '# to be #-th training pose')
    parser.add_argument("--t", type=str, default='',
                        help='render time control, empty to be arange(len(views)), '
                             '#,#,# to be #-th frame, #:# to be #(include) to #(exclude) frame, :, can be mixed')
    parser.add_argument("--f", type=int, default=-1,
                        help='overwrite the frame number when loading the render pose')
    parser.add_argument("--type", type=str, default='vid',
                        help='choose among seq, vid, depth, depthrgb')
    parser.add_argument("--render_scaling", type=float, default=1,
                        help='radius of the render spire')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    evaluate(args)

