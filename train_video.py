import os
import torch
import imageio
import numpy as np
import math
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn
from VIDEO import *
from tqdm import tqdm, trange
from dataloader import load_videos
import shutil
from datetime import datetime
from metrics import compute_img_metric
import cv2
import configargparse

torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    parser = config_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # set up multi-processing
    if args.gpu_num == -1:
        args.gpu_num = torch.cuda.device_count()
        print(f"Using {args.gpu_num} GPU(s)")

    print(f"Training: {args.expname}")
    images = load_videos(args.datadir, args.factor)

    T = len(images)
    H, W = images[0].shape[0:2]
    print('Loaded video', T, H, W)

    # Resove pyramid related configs
    if args.pyr_minimal_dim < 0:
        # store the iter_num when starting the stage
        pyr_stages = list(map(int, args.pyr_stage.split(','))) if len(args.pyr_stage) > 0 else []
        pyr_stages = [0] + pyr_stages  # one default stage
        pyr_levelis = list(range(len(pyr_stages)))[::-1]
        pyr_factors = [args.pyr_factor ** i for i in pyr_levelis]
    else:
        raise NotImplementedError()  # TODO: advanced pyramid config
    print("Pyramid info: ")
    for leveli, stage, factor in zip(pyr_levelis, pyr_stages, pyr_factors):
        print(f"    level {leveli}: start in {stage} iter, with factor {factor}")

    # Summary writers
    writer = SummaryWriter(os.path.join(args.expdir, args.expname))

    # Create nerf model
    nerf = SimpleVideo(args, H, W, T)
    nerf = nn.DataParallel(nerf, list(range(args.gpu_num)))

    optimizer = torch.optim.Adam(params=nerf.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(params=nerf.parameters(), lr=args.lrate, momentum=0.9)

    ##########################
    # Load checkpoints
    ckpts = [os.path.join(args.expdir, args.expname, f)
             for f in sorted(os.listdir(os.path.join(args.expdir, args.expname))) if 'tar' in f]
    print('Found ckpts', ckpts)

    start = 0
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        smart_load_state_dict(nerf, ckpt)

    global_step = start

    print('Begin training')

    start = start + 1
    N_iters = args.N_iters + 1

    def generate_dataset(_imgs):
        H, W = _imgs.shape[-2:]
        # generate patch information
        patch_h_start = np.arange(0, H, args.patch_h_stride)
        patch_w_start = np.arange(0, W, args.patch_w_stride)

        patch_wh_start = np.meshgrid(patch_h_start, patch_w_start)
        patch_wh_start = np.stack(patch_wh_start[::-1], axis=-1).reshape(-1, 2)[None, ...]

        patch_wh_start = patch_wh_start.reshape(-1, 2)
        patch_wh_start = torch.tensor(patch_wh_start)

        H_pad = patch_h_start.max() + patch_h_size - H
        W_pad = patch_w_start.max() + patch_w_size - W

        _imgs_pad = torchf.pad(_imgs, [0, W_pad, 0, H_pad])
        return patch_wh_start, _imgs_pad

    patch_h_size = args.patch_h_size
    patch_w_size = args.patch_w_size
    # start training
    pbar = trange(start, N_iters)
    for i in pbar:
        # pyramid processing, including initialization
        if i in pyr_stages or i == start:
            # leveli = pyr_stages.index(global_step)
            leveli = np.searchsorted(pyr_stages, global_step, 'right') - 1
            pyr_factor = pyr_factors[leveli]

            target_images = [cv2.resize(img, None, None, pyr_factor, pyr_factor, interpolation=cv2.INTER_AREA)
                             for img in images]
            target_images = torch.stack([torch.tensor(img) for img in target_images]) / 255
            target_images = target_images.permute(0, 3, 1, 2)
            patch_wh_start, target_images_pad = generate_dataset(target_images)

            # initialize
            len_data = len(patch_wh_start)
            i_batch = 0
            permute = np.random.permutation(len_data)
            print(f"Start level {leveli} of factor {pyr_factor}, generate {len_data} data")

        # random shuffle dataset
        if i_batch >= len_data:
            permute = np.random.permutation(len_data)
            i_batch = 0

        data_indice = permute[i_batch:i_batch + 1]
        i_batch += 1

        w_start, h_start = torch.split(patch_wh_start[data_indice], 1, dim=-1)
        w_start, h_start = w_start[..., 0], h_start[..., 0]

        b_rgbs = torch.stack([target_images_pad[:, :, hs: hs + patch_h_size, ws: ws + patch_w_size]
                              for hs, ws in zip(h_start, w_start)]).permute(0, 2, 1, 3, 4)

        #####  Core optimization loop  #####
        nerf.train()
        if hasattr(nerf.module, "update_step"):
            nerf.module.update_step(global_step)

        rgb, extra = nerf(patch_h_size, patch_w_size, h_start, w_start, res=b_rgbs)

        swd_loss = extra.pop("swd").mean()
        # define extra losses here
        args_var = vars(args)
        extra_losses = {}
        for k, v in extra.items():
            if args_var[f"{k}_loss_weight"] > 0:
                extra_losses[k] = extra[k].mean() * args_var[f"{k}_loss_weight"]

        loss = swd_loss
        for v in extra_losses.values():
            loss = loss + v

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        ################################

        if i % args.i_img == 0:
            writer.add_scalar('weight/lr', new_lrate, i)
            writer.add_scalar('aloss/swd', swd_loss.item(), i)
            for k, v in extra.items():
                writer.add_scalar(f'{k}', float(v.mean()), i)

        if i % args.i_print == 0:
            pbar.set_description(f"[TRAIN] Iter: {i} Loss: {loss.item():.4f} SWD: {swd_loss.item():.4f}" +
                                 "|".join([f"{k}: {v.item():.4f}" for k, v in extra_losses.items()]))

        if i % args.i_weights == 0:
            path = os.path.join(args.expdir, args.expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_state_dict': nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if args.optimize_poses:
                save_dict['rot_raw'] = rot_raw
                save_dict['tran_raw'] = tran_raw
                save_dict['intrin_raw'] = intrin_raw
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0:
            pass
            # TODO

        if i % args.i_eval == 0:
            pass
            # TODO

        if i % args.i_video == 0:
            moviebase = os.path.join(args.expdir, args.expname, f'{i:06d}_')
            nerf.module.export_video(moviebase)

        global_step += 1


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--gpu_num", type=int, default='-1', help='number of processes')
    parser.add_argument("--datadir", type=str,
                        help='input data directory')
    parser.add_argument("--expdir", type=str,
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--seed", type=int, default=666,
                        help='random seed')
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')

    parser.add_argument("--model_type", type=str, default="Video",
                        help='model type')
    parser.add_argument("--rgb_activate", type=str, default='none',
                        help='activate function for rgb output, choose among "none", "sigmoid"')

    # pyramid configuration
    parser.add_argument("--pyr_stage", type=str, default='',
                        help='x,y,z,...   iteration to upsample')
    parser.add_argument("--pyr_minimal_dim", type=int, default=-1,  # TODO: implement this
                        help='if > 0, will determine the pyr_stage')
    parser.add_argument("--pyr_iter", type=int, default=-1,
                        help='iter num in each level')
    parser.add_argument("--pyr_factor", type=float, default=0.5,
                        help='factor in each pyr level')

    # for mpi, not use for now
    parser.add_argument("--mpi_h_scale", type=float, default=1.4,
                        help='the height of the stored MPI is <mpi_h_scale * H>')
    parser.add_argument("--mpi_w_scale", type=float, default=1.4,
                        help='the width of the stored MPI is <mpi_w_scale * W>')
    parser.add_argument("--mpi_h_verts", type=int, default=12,
                        help='the height of the stored MPI is <mpi_h_scale * H>')
    parser.add_argument("--mpi_w_verts", type=int, default=15,
                        help='the width of the stored MPI is <mpi_w_scale * W>')
    parser.add_argument("--mpi_d", type=int, default=64,
                        help='number of the MPI layer')
    parser.add_argument("--atlas_grid_h", type=int, default=8,
                        help='atlas_grid_h * atlas_grid_w == mpi_d')
    parser.add_argument("--atlas_size_scale", type=float, default=1,
                        help='atlas_size = mpi_d * H * W * atlas_size_scale')
    parser.add_argument("--atlas_cnl", type=int, default=4,
                        help='channel num')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    # training options
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'],
                        help='optmizer')
    parser.add_argument("--patch_h_size", type=int, default=512,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--patch_w_size", type=int, default=512,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--patch_w_stride", type=int, default=-1,
                        help='patch stride, if < 0, automatically decide')
    parser.add_argument("--patch_h_stride", type=int, default=-1,
                        help='patch stride, if < 0, automatically decide')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=30,
                        help='exponential learning rate decay (in 1000 steps)')

    # loss related
    parser.add_argument("--swd_patch_size", type=int, default=7,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--swd_patcht_size", type=int, default=7,
                        help='exponential learning rate decay (in 1000 steps)')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=50000)
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')

    # logging options
    parser.add_argument("--i_img",    type=int, default=300,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_print",   type=int, default=300,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=20000,
                        help='frequency of testset saving')
    parser.add_argument("--i_eval", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')
    return parser


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
