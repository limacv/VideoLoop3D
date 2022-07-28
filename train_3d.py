import os
import torch
import imageio
import numpy as np
import math
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from MPI import *

from dataloader import load_llff_data, poses_avg
from utils import *
import shutil
from datetime import datetime
from metrics import compute_img_metric
import cv2
import configargparse

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
                                             fps=30, quality=8)

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
                    imageio.mimwrite(moviebase + f'rgb{suffix}.mp4', to8b(rgbs), fps=30, quality=8)
                    imageio.mimwrite(moviebase + f'disp{suffix}.mp4', to8b(disps / np.max(disps)), fps=30,
                                     quality=8)
            return

            # Prepare raybatch tensor if batching random rays
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


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--datadir", type=str,
                        help='input data directory')
    parser.add_argument("--expdir", type=str,
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--seed", type=int, default=666,
                        help='random seed')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # for mpi
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
    parser.add_argument("--model_type", type=str, default="MPI",
                        choices=["MPI", "MPMesh"])
    parser.add_argument("--rgb_mlp_type", type=str, default='direct',
                        help='mlp type, choose among "direct", "rgbamlp", "rgbmlp"')
    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--optimize_depth", action='store_true',
                        help='if true, optimzing the depth of each plane')
    parser.add_argument("--optimize_normal", action='store_true',
                        help='if true, optimzing the normal of each plane')
    parser.add_argument("--optimize_geo_start", type=int, default=100000,
                        help='iteration to start optimizing verts and uvs')
    parser.add_argument("--optimize_verts_gain", type=float, default=1,
                        help='set 0 to disable the vertices optimization')
    parser.add_argument("--optimize_uvs_gain", type=float, default=1,
                        help='set 0 to disable the uvs optimization')
    parser.add_argument("--normalize_verts", action='store_true',
                        help='if true, the parameter is normalized')

    # about training
    parser.add_argument("--upsample_stage", type=str, default="",
                        help='x,y,z,...  stage to perform upsampling')
    parser.add_argument("--rgb_smooth_loss_weight", type=float, default=0,
                        help='rgb smooth loss')
    parser.add_argument("--a_smooth_loss_weight", type=float, default=0,
                        help='rgb smooth loss')
    parser.add_argument("--d_smooth_loss_weight", type=float, default=0,
                        help='depth smooth loss')
    parser.add_argument("--laplacian_loss_weight", type=float, default=0,
                        help='as rigid as possible smooth loss')

    # training options
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'],
                        help='optmizer')
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--patch_h_size", type=int, default=512,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--patch_w_size", type=int, default=512,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=30,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=50000)
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,
                        help='number of additional fine samples per ray')
    parser.add_argument("--N_samples_fine", type=int, default=64,
                        help='n sample fine = N_samples_fine + N_importance')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", type=bool, default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--embed_type", type=str, default='pe',
                        help='pe, none, hash, dict')
    parser.add_argument("--log2_embed_hash_size", type=int, default=19,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_window_start", type=int, default=0,
                        help='windowed PE start step')
    parser.add_argument("--multires_window_end", type=int, default=-1,
                        help='windowed PE end step, negative to disable')
    parser.add_argument("--multires_views_window_start", type=int, default=0,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_views_window_end", type=int, default=-1,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1e0,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_rgba", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_texture", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_view", type=int, default=-1,
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_slice", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_keypoints", action='store_true',
                        help='change the keypoint location')
    parser.add_argument("--render_deformed", type=str, default='',
                        help='edited file')
    parser.add_argument("--render_factor", type=float, default=1,
                        help='change the keypoint location')
    parser.add_argument("--render_canonical", action='store_true',
                        help='if true, the DNeRF is like traditional NeRF')

    ## data options
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

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

    # multiprocess learning
    parser.add_argument("--gpu_num", type=int, default='-1', help='number of processes')

    # test use latent_t
    parser.add_argument("--render_chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # MA Li in General
    # ===================================
    parser.add_argument("--frm_num", type=int, default=-1, help='number of frames to use')
    parser.add_argument("--bd_factor", type=float, default=0.65, help='expand factor of the ROI box')
    parser.add_argument("--optimize_poses", default=False, action='store_true',
                        help='optimize poses')
    parser.add_argument("--optimize_poses_start", type=int, default=0, help='start step of optimizing poses')
    parser.add_argument("--surpress_boundary_thickness", type=int, default=0,
                        help='do not supervise the boundary of thickness <>, 0 to disable')
    parser.add_argument("--itertions_per_frm", type=int, default=50)
    parser.add_argument("--masked_sample_precent", type=float, default=0.92,
                        help="in batch_size samples, precent of the samples that are sampled at"
                             "masked region, set to 1 to disable the samples on black region")
    parser.add_argument("--sigma_activate", type=str, default='relu',
                        help='activate function for sigma output, choose among "relu", "softplus",'
                             '"volsdf"')
    parser.add_argument("--use_raw2outputs_old", type=bool, default=True,
                        help='use the original raw2output (not forcing the last layer to be alpha=1')
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='if true, nerf_coarse == nerf_fine')
    parser.add_argument("--not_supervise_rgb0", action='store_true', default=False,
                        help='if true, rgb0 well not considered as part of the loss')
    parser.add_argument("--best_frame_idx", type=int, default=-1,
                        help='if > 0, the first epoch will be trained only on this frame')

    ## For other losses
    parser.add_argument("--sparsity_type", type=str, default='none',
                        help='sparsity loss type, choose among none, l1, l1/l2, entropy')
    parser.add_argument("--sparsity_loss_weight", type=float, default=0,
                        help='sparsity loss weight')
    parser.add_argument("--sparsity_loss_start_step", type=float, default=50000,
                        help='sparsity loss weight')
    parser.add_argument("--use_two_time_for_cycle", type=bool, default=False,
                        help='pe or latent')
    return parser


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
