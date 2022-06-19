import os
import torch
import imageio
import numpy as np
import math
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
from MPI import *

from configs import config_parser
from dataloader import load_llff_data, poses_avg
from utils import *
import shutil
from datetime import datetime
from metrics import compute_img_metric
import cv2

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

    images, poses, intrins, bds, render_poses, render_intrins = load_llff_data(basedir=args.datadir,
                                                                               factor=args.factor,
                                                                               recenter=True)

    H, W = images[0].shape[0:2]
    V = len(images)
    print('Loaded llff', V, H, W, poses.shape, intrins.shape, render_poses.shape, bds.shape)

    ref_pose = poses_avg(poses)[:, :4]
    ref_extrin = pose2extrin_np(ref_pose)
    ref_intrin = intrins[0]
    ref_near, ref_far = bds[:, 0].min(), bds[:, 1].max()

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
    optimizer = torch.optim.Adam(params=nerf.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(params=nerf.parameters(), lr=args.lrate, momentum=0.9)

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

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():

            i = start + 1
            if hasattr(nerf.module, "update_step"):
                nerf.module.update_step(i)
            suffix = ''
            if len(args.texture_map_post) > 0:
                assert os.path.isfile(args.texture_map_post)
                print(f'loading texture map from {args.texture_map_post}')
                nerf.module.force_load_texture_map(args.texture_map_post, args.texture_map_post_isfull, args.texture_map_force_map)
                suffix += '_' + os.path.basename(args.texture_map_post).split('.')[0]

            if len(args.geometry_map_post) > 0:
                assert os.path.isfile(args.geometry_map_post)
                print(f'loading geometry map from {args.geometry_map_post}')
                nerf.module.force_load_geometry_map(args.geometry_map_post, args.geometry_map_post_isfull)
                suffix += '_' + os.path.basename(args.geometry_map_post).split('.')[0]

            savedir = os.path.join(args.expdir, args.expname, f'render_only')
            os.makedirs(savedir, exist_ok=True)

            if args.render_view >= 0:
                render_poses = poses[args.render_view:args.render_view+1].expand(T, -1, -1)
                render_intrins = intrins[args.render_view:args.render_view+1].expand(T, -1, -1)
                suffix += f"_view{args.render_view:03d}"

            print('render fix t: ', render_poses.shape, render_intrins.shape)
            dummy_num = ((len(render_poses) - 1) // args.gpu_num + 1) * args.gpu_num - len(render_poses)
            dummy_poses = torch.eye(3, 4).unsqueeze(0).expand(dummy_num, 3, 4).type_as(render_poses)
            dummy_intrinsic = render_intrins[:dummy_num].clone()
            print(f"Append {dummy_num} # of poses to fill all the GPUs")

            with torch.no_grad():
                nerf.eval()
                render_time = 0

                if args.render_texture:
                    print('saving texture map')
                    if T <= 1:
                        texture_maps = nerf.module.get_texture_map()
                        texture_maps = [tex[0, :3].detach().permute(1, 2, 0).cpu().numpy() for tex in texture_maps]
                        for i, texture_map in enumerate(texture_maps):
                            imageio.imwrite(savedir + f"/texture{suffix}_{i}.png",
                                            to8b(texture_map))
                    else:
                        texture_maps = []
                        for ti in range(T):
                            print(f"Get texture map {ti}")
                            texture_map = nerf.module.get_texture_map(t=ti)
                            texture_map = [to8b(tex[0, :3].detach().permute(1, 2, 0).cpu().numpy()) for tex in
                                           texture_map]
                            texture_maps.append(texture_map)
                        for i in range(len(texture_maps[0])):
                            texture_map = [ts[i] for ts in texture_maps]
                            imageio.mimwrite(savedir + f"/texture_{suffix}_{i}.mp4", texture_map,
                                             fps=30, quality=10)

                if T > 1 and not args.render_test:
                    rH, rW = H * args.render_factor, W * args.render_factor
                    render_intrins = render_intrins.clone()
                    render_intrins[:, :2, :3] *= args.render_factor
                    rgbs, disps = nerf(rH, rW, chunk=args.render_chunk,
                                       poses=torch.cat([render_poses, dummy_poses], dim=0),
                                       intrins=torch.cat([render_intrins, dummy_intrinsic], dim=0),
                                       render_kwargs=render_kwargs_test)
                    rgbs = rgbs[:len(rgbs) - dummy_num]
                    disps = disps[:len(disps) - dummy_num]
                    disps = (disps - disps.min()) / (disps.max() - disps.min()).clamp_min(1e-10)
                    rgbs = rgbs.cpu().numpy()
                    disps = disps.cpu().numpy()
                    moviebase = os.path.join(savedir, f'{args.expname}_varyt_{render_time:06d}_')
                    imageio.mimwrite(moviebase + f'rgb{suffix}.mp4', to8b(rgbs), fps=30, quality=10)
                    imageio.mimwrite(moviebase + f'disp{suffix}.mp4', to8b(disps / np.max(disps)), fps=30,
                                     quality=10)
            return

            # Prepare raybatch tensor if batching random rays
    print('Begin', args.batch_size)

    start = start + 1
    N_iters = args.N_iters + 1
    batch_size = args.batch_size

    # generate patch information
    patch_h_start = np.arange(0, H, batch_size)
    patch_w_start = np.arange(0, W, batch_size)

    patch_wh_start = np.meshgrid(patch_h_start, patch_w_start)
    patch_wh_start = np.stack(patch_wh_start[::-1], axis=-1).reshape(-1, 2)[None, ...]
    patch_wh_start = np.repeat(patch_wh_start, V, axis=0)

    view_index = np.arange(V)[:, None, None].repeat(patch_wh_start.shape[1], axis=0)

    patch_wh_start = patch_wh_start.reshape(-1, 2)
    view_index = view_index.reshape(-1)
    len_data = len(patch_wh_start)

    patch_wh_start = torch.tensor(patch_wh_start)
    view_index = torch.tensor(view_index).long()

    H_pad = patch_h_start.max() + batch_size - H
    W_pad = patch_w_start.max() + batch_size - W
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
        b_rgbs = torch.stack([images[v, :, hs: hs + batch_size, ws: ws + batch_size]
                              for v, hs, ws in zip(view_idx, h_start, w_start)])

        #####  Core optimization loop  #####
        nerf.train()
        if hasattr(nerf.module, "update_step"):
            nerf.module.update_step(global_step)
        rgb, extra = nerf(batch_size, batch_size, b_extrin, b_intrin_patch)

        # RGB loss
        img_loss = img2mse(rgb, b_rgbs)
        psnr = mse2psnr(img_loss)

        # define extra losses here
        if args.sparsity_loss_weight > 0:
            sparsity_loss = extra["sparsity"].mean()
            sparsity_weight_apply = args.sparsity_loss_weight
        else:
            sparsity_loss, sparsity_weight_apply = 0, 0

        loss = img_loss \
                + sparsity_loss * sparsity_weight_apply

        optimizer.zero_grad()
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            pose_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.optimize_poses and global_step >= args.optimize_poses_start:
            pose_optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        ################################

        if i % args.i_img == 0:
            writer.add_scalar('aloss/psnr', psnr, i)
            writer.add_scalar('aloss/mse_loss', loss, i)
            writer.add_scalar('aloss/sparsity_loss', sparsity_loss, i)
            writer.add_scalar('weight/lr', new_lrate, i)

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()}")

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
                imageio.mimwrite(moviebase + '_rgb.mp4', to8b(rgbs), fps=30, quality=10)

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
