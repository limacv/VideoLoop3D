import os
import torch
import imageio
import numpy as np
import math
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from MPI import *

from dataloader import load_mv_videos, poses_avg, load_llff_data
from utils import *
import shutil
from datetime import datetime
from metrics import compute_img_metric
import cv2
from config_parser import config_parser

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
    images, poses, intrins, bds, render_poses, render_intrins = load_llff_data(basedir=args.datadir,
                                                                               factor=args.factor,
                                                                               bd_factor=args.bd_factor,
                                                                               recenter=True)

    H, W = images[0].shape[0:2]
    V = len(images)
    print('Loaded llff', V, H, W, poses.shape, intrins.shape, render_poses.shape, bds.shape)

    ref_pose = poses_avg(poses)[:, :4]
    ref_extrin = pose2extrin_np(ref_pose)
    ref_intrin = intrins[0]
    ref_near, ref_far = bds[:, 0].min(), bds[:, 1].max()

    # resolve scheduling
    upsample_stage = list(map(int, args.upsample_stage.split(','))) if len(args.upsample_stage) > 0 else []
    args.upsample_stage = upsample_stage

    # Summary writers
    writer = SummaryWriter(os.path.join(args.expdir, args.expname))

    # Create log dir and copy the config file
    if not args.render_only:
        file_path = os.path.join(args.expdir, args.expname, f"source_{datetime.now().timestamp():.0f}")
        os.makedirs(file_path, exist_ok=True)
        f = os.path.join(file_path, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(file_path, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())
        files_copy = [fs for fs in os.listdir(".") if ".py" in fs]
        for fc in files_copy:
            shutil.copyfile(f"./{fc}", os.path.join(file_path, fc))

    # Create nerf model
    if args.model_type == "MPI":
        nerf = MPI(args, H, W, ref_extrin, ref_intrin, ref_near, ref_far)
    elif args.model_type == "MPMesh":
        nerf = MPMesh(args, H, W, ref_extrin, ref_intrin, ref_near, ref_far)
    else:
        raise RuntimeError(f"Unrecognized model type {args.model_type}")

    nerf = nn.DataParallel(nerf, list(range(args.gpu_num)))
    if hasattr(nerf.module, "get_optimizer"):
        print(f"Using {type(nerf)}'s get_optimizer()")
        optimizer = nerf.module.get_optimizer()
    else:
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params=nerf.parameters(), lr=args.lrate, betas=(0.9, 0.999))
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params=nerf.parameters(), lr=args.lrate, momentum=0.9)
        else:
            raise RuntimeError(f"Unrecongnized optimizer type {args.optimizer}")

    render_extrins = pose2extrin_np(render_poses)
    render_extrins = torch.tensor(render_extrins).float()
    render_intrins = torch.tensor(render_intrins).float()

    ######################
    # if optimize poses
    poses = torch.tensor(poses)
    intrins = torch.tensor(intrins)
    if args.optimize_poses:
        rot_raw = poses[:, :3, :2]
        tran_raw = poses[:, :3, 3]
        intrin_raw = intrins[:, :2, :3]

        # leave the first pose unoptimized
        rot_raw0, tran_raw0, intrin_raw0 = rot_raw[:1], tran_raw[:1], intrin_raw[:1]
        rot_raw = nn.Parameter(rot_raw[1:], requires_grad=True)
        tran_raw = nn.Parameter(tran_raw[1:], requires_grad=True)
        intrin_raw = nn.Parameter(intrin_raw[1:], requires_grad=True)
        pose_optimizer = torch.optim.SGD(params=[rot_raw, tran_raw, intrin_raw],
                                         lr=args.lrate / 5)
    else:
        rot_raw0, tran_raw0, intrin_raw0 = None, None, None
        rot_raw, tran_raw, intrin_raw = None, None, None
        pose_optimizer = None

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
        if 'rot_raw' in ckpt.keys():
            print("Loading poses and intrinsics from the ckpt")
            rot_raw = ckpt['rot_raw']
            tran_raw = ckpt['tran_raw']
            intrin_raw = ckpt['intrin_raw']
            poses, intrinsics = raw2poses(
                torch.cat([rot_raw0, rot_raw]),
                torch.cat([tran_raw0, tran_raw]),
                torch.cat([intrin_raw0, intrin_raw]))
            assert len(rot_raw) + 1 == V

    global_step = start

    print('Begin')

    start = start + 1
    N_iters = args.N_iters + 1
    patch_h_size = args.patch_h_size
    patch_w_size = args.patch_w_size

    # generate patch information
    patch_h_start = np.arange(0, H, patch_h_size)
    patch_w_start = np.arange(0, W, patch_w_size)

    patch_wh_start = np.meshgrid(patch_h_start, patch_w_start)
    patch_wh_start = np.stack(patch_wh_start[::-1], axis=-1).reshape(-1, 2)[None, ...]
    patch_wh_start = np.repeat(patch_wh_start, V, axis=0)

    view_index = np.arange(V)[:, None, None].repeat(patch_wh_start.shape[1], axis=0)

    patch_wh_start = patch_wh_start.reshape(-1, 2)
    view_index = view_index.reshape(-1)
    len_data = len(patch_wh_start)

    patch_wh_start = torch.tensor(patch_wh_start)
    view_index = torch.tensor(view_index).long()

    H_pad = patch_h_start.max() + patch_h_size - H
    W_pad = patch_w_start.max() + patch_w_size - W
    images = torch.tensor(images)
    images = torchf.pad(images.permute(0, 3, 1, 2), [0, W_pad, 0, H_pad])

    # start training
    permute = np.random.permutation(len_data)
    i_batch = 0
    for i in range(start, N_iters):
        if i_batch >= len_data:
            permute = np.random.permutation(len_data)
            i_batch = 0

        data_indice = permute[i_batch:i_batch + 1]
        i_batch += 1
        w_start, h_start = torch.split(patch_wh_start[data_indice], 1, dim=-1)
        w_start, h_start = w_start[..., 0], h_start[..., 0]
        view_idx = view_index[data_indice]

        # if optimizing camera poses, regenerating the rays
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            poses, intrins = raw2poses(
                torch.cat([rot_raw0, rot_raw]),
                torch.cat([tran_raw0, tran_raw]),
                torch.cat([intrin_raw0, intrin_raw]))

        b_pose = poses[view_idx]
        b_extrin = pose2extrin_torch(b_pose)
        b_intrin = intrins[view_idx]
        b_intrin_patch = get_new_intrin(b_intrin, h_start, w_start).float()
        b_rgbs = torch.stack([images[v, :, hs: hs + patch_h_size, ws: ws + patch_w_size]
                              for v, hs, ws in zip(view_idx, h_start, w_start)])

        #####  Core optimization loop  #####
        nerf.train()
        if hasattr(nerf.module, "update_step"):
            nerf.module.update_step(global_step)

        rgb, extra = nerf(patch_h_size, patch_w_size, b_extrin, b_intrin_patch)

        # RGB loss
        img_loss = img2mse(rgb, b_rgbs)
        psnr = mse2psnr(img_loss)

        # define extra losses here
        args_var = vars(args)
        extra_losses = {}
        for k, v in extra.items():
            if args_var[f"{k}_loss_weight"] > 0:
                extra_losses[k] = extra[k].mean() * args_var[f"{k}_loss_weight"]

        loss = img_loss
        for v in extra_losses.values():
            loss = loss + v

        # old_vert = nerf.module.verts.detach().clone()
        optimizer.zero_grad()
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            pose_optimizer.zero_grad()
        loss.backward()
        if hasattr(nerf.module, "post_backward"):
            nerf.module.post_backward()
        optimizer.step()
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            pose_optimizer.step()
        # delta_vert = (nerf.module.verts.detach() - old_vert).abs().max()
        # print(f"max delta vert = {delta_vert.item()}")

        ###   update learning rate   ###
        if hasattr(nerf.module, "get_lrate"):
            name_lrates = nerf.module.get_lrate(global_step)
        else:
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            name_lrates = [("lr", new_lrate)] * len(optimizer.param_groups)

        for (lrname, new_lrate), param_group in zip(name_lrates, optimizer.param_groups):
            param_group['lr'] = new_lrate

        ################################

        if i % args.i_img == 0:
            writer.add_scalar('aloss/psnr', psnr, i)
            writer.add_scalar('aloss/mse_loss', loss, i)
            for k, v in extra.items():
                writer.add_scalar(f'{k}', float(v.mean()), i)
            for name, newlr in name_lrates:
                writer.add_scalar(f'lr/{name}', newlr, i)

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item():.4f} PSNR: {psnr.item():.4f}",
                  "|".join([f"{k}: {v.item():.4f}" for k, v in extra_losses.items()]))

        if i % args.i_weights == 0:
            if hasattr(nerf.module, "save_mesh"):
                prefix = os.path.join(args.expdir, args.expname, f"mesh{i:06d}")
                nerf.module.save_mesh(prefix)

            if hasattr(nerf.module, "save_texture"):
                prefix = os.path.join(args.expdir, args.expname, f"texture{i:06d}")
                nerf.module.save_texture(prefix)

            # path = os.path.join(args.expdir, args.expname, '{:06d}.tar'.format(i))
            # save_dict = {
            #     'global_step': global_step,
            #     'network_state_dict': nerf.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }
            # if args.optimize_poses:
            #     save_dict['rot_raw'] = rot_raw
            #     save_dict['tran_raw'] = tran_raw
            #     save_dict['intrin_raw'] = intrin_raw
            # torch.save(save_dict, path)
            # print('Saved checkpoints at', path)

        if i % args.i_testset == 0:
            pass
            # TODO

        if i % args.i_eval == 0:
            pass
            # TODO

        if i % args.i_video == 0:
            moviebase = os.path.join(args.expdir, args.expname, f'{i:06d}_')
            print('render poses shape', render_extrins.shape, render_intrins.shape)
            with torch.no_grad():
                nerf.eval()

                rgbs = []
                for ri in range(len(render_extrins)):
                    r_pose = render_extrins[ri:ri + 1]
                    r_intrin = render_intrins[ri:ri + 1]

                    rgb, extra = nerf(H, W, r_pose, r_intrin)
                    rgb = rgb[0].permute(1, 2, 0).cpu().numpy()
                    rgbs.append(rgb)

                rgbs = np.array(rgbs)
                imageio.mimwrite(moviebase + '_rgb.mp4', to8b(rgbs), fps=30, quality=8)

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
