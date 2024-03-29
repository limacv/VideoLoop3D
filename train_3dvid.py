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


class MVVidPatchDataset(Dataset):
    def __init__(self, resize_hw, videos, patch_size, patch_stride, poses, intrins,
                 loss_configs=None):
        super().__init__()
        h_raw, w_raw, _ = videos[0][0].shape[-3:]
        self.h, self.w = resize_hw
        self.v = len(videos)
        self.poses = poses.clone().cpu()
        self.intrins = intrins.clone().cpu()
        self.intrins[:, :2] *= torch.tensor([self.w / w_raw, self.h / h_raw]).reshape(1, 2, 1).type_as(intrins)
        self.patch_h_size, self.patch_w_size = patch_size
        if self.h * self.w < self.patch_h_size * self.patch_w_size:
            patch_wh_start = torch.tensor([[0, 0]]).long().reshape(-1, 2)
            pad_info = [0, 0, 0, 0]
            self.patch_h_size, self.patch_w_size = self.h, self.w
        else:
            patch_wh_start, pad_info = generate_patchinfo(self.h, self.w, patch_size, patch_stride)

        patch_wh_start = patch_wh_start[None, ...].expand(self.v, -1, 2)
        view_index = np.arange(self.v)[:, None, None].repeat(patch_wh_start.shape[1], axis=1)
        self.patch_wh_start = patch_wh_start.reshape(-1, 2).cpu()
        self.view_index = view_index.reshape(-1).tolist()
        self.loss_configs = loss_configs
        assert len(self.loss_configs) == self.v

        self.videos = []
        for video in videos:
            vid = np.array([cv2.resize(img, (self.w, self.h)) for img in video])
            vid = torch.tensor(vid, device='cpu')
            vid = (vid / 255).permute(0, 3, 1, 2)
            vid = torchf.pad(vid, pad_info)
            self.videos.append(vid)
        print(f"Dataset: generate {len(self)} patches for training, pad {pad_info} to videos")

    def __len__(self):
        return len(self.patch_wh_start)

    def __getitem__(self, item):
        w_start, h_start = self.patch_wh_start[item]
        view_idx = self.view_index[item]
        pose = self.poses[view_idx]
        intrin = get_new_intrin(self.intrins[view_idx], h_start, w_start).float()
        crops = self.videos[view_idx][..., h_start: h_start + self.patch_h_size, w_start: w_start + self.patch_w_size]
        cfg = deepcopy(self.loss_configs[view_idx])
        return w_start, h_start, pose, intrin, crops, cfg


def train(args):
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
    test_view = args.test_view_idx
    test_view = list(map(int, test_view.split(','))) if len(test_view) > 0 else []
    train_view = sorted(list(set(range(V)) - set(test_view)))
    # filter out test view
    videos = [videos[train_i] for train_i in train_view]
    poses = poses[train_view]
    intrins = intrins[train_view]
    print(f'Training view: {train_view}')

    ref_pose = poses_avg(poses)[:, :4]
    ref_extrin = pose2extrin_np(ref_pose)
    ref_intrin = intrins[0]
    ref_near, ref_far = bds.min(), bds.max()

    # Resove pyramid related configs, controled by (pyr_stage, pyr_factor, N_iters)
    #                                           or (pyr_minimal_dim, pyr_factor, pyr_num_epoch)
    if args.pyr_minimal_dim < 0:
        # store the iter_num when starting the stage
        pyr_stages = list(map(int, args.pyr_stage.split(','))) if len(args.pyr_stage) > 0 else []
        pyr_stages = np.array([0] + pyr_stages + [args.N_iters])  # one default stage
        pyr_num_epoch = pyr_stages[1:] - pyr_stages[:-1]
        pyr_factors = [args.pyr_factor ** i for i in list(range(len(pyr_num_epoch)))[::-1]]
        pyr_hw = [(int(H * f), int(W * f)) for f in pyr_factors]
    else:
        num_stage = int(np.log(args.pyr_minimal_dim / min(H, W)) / np.log(args.pyr_factor)) + 1
        pyr_factors = [args.pyr_factor ** i for i in list(range(num_stage))[::-1]]
        pyr_hw = [(int(H * f), int(W * f)) for f in pyr_factors]
        pyr_num_epoch = [args.pyr_num_epoch] * num_stage
    print("Pyramid info: ")
    for leveli, (f_, hw_, num_step_) in enumerate(zip(pyr_factors, pyr_hw, pyr_num_epoch)):
        print(f"    level {leveli}: factor {f_} [{hw_[0]} x {hw_[1]}] run for {num_step_} iterations")
    # end of pyramid infomation

    # Summary writers
    writer = SummaryWriter(os.path.join(expdir, expname))

    # Create log dir and copy the config file
    file_path = os.path.join(expdir, expname, f"source_{datetime.now().timestamp():.0f}")
    os.makedirs(file_path, exist_ok=True)
    f = os.path.join(file_path, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if len(args.config) > 0:
        f = os.path.join(file_path, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    if len(args.config1) > 0:
        f = os.path.join(file_path, 'config1.txt')
        with open(f, 'w') as file:
            file.write(open(args.config1, 'r').read())
    files_copy = [fs for fs in os.listdir(".") if ".py" in fs]
    for fc in files_copy:
        shutil.copyfile(f"./{fc}", os.path.join(file_path, fc))

    # Create nerf model
    if args.model_type == "MPMesh":
        nerf = MPMeshVid(args, H, W, ref_extrin, ref_intrin, ref_near, ref_far)
    else:
        raise RuntimeError(f"Unrecognized model type {args.model_type}")

    nerf = DataParallelCPU(nerf) if device == 'cpu' else nn.DataParallel(nerf, list(range(args.gpu_num)))
    nerf.to(device)
    render_extrins = pose2extrin_np(render_poses)
    render_extrins = torch.tensor(render_extrins).float()
    render_intrins = torch.tensor(render_intrins).float()

    poses = torch.tensor(poses)
    intrins = torch.tensor(intrins)

    # figuring out the loss config
    loss_config_other = {
        "loss_name": args.loss_name,
        "patch_size": args.swd_patch_size,
        "patcht_size": args.swd_patcht_size,
        "stride": args.swd_stride,
        "stridet": args.swd_stridet,
        "alpha": args.swd_alpha,
        "rou": args.swd_rou,
        "scaling": args.swd_scaling,
        "dist_fn": args.swd_dist_fn,
        "macro_block": args.swd_macro_block,
        "factor": args.swd_factor,
    }
    loss_config_ref = {
        "loss_name": args.loss_name_ref,
        "loss_gain": args.swd_loss_gain_ref,
        "patch_size": args.swd_patch_size_ref,
        "patcht_size": args.swd_patcht_size_ref,
        "stride": args.swd_stride_ref,
        "stridet": args.swd_stridet_ref,
        "alpha": args.swd_alpha_ref,
        "rou": args.swd_rou_ref,
        "scaling": args.swd_scaling_ref,
        "dist_fn": args.swd_dist_fn_ref,
        "macro_block": args.swd_macro_block,
        "factor": args.swd_factor_ref,
    }
    loss_cfgs = [loss_config_other] * V
    ref_idxs = list(map(int, args.loss_ref_idx.split(',')))
    for ref_idx in ref_idxs:
        loss_cfgs[ref_idx] = loss_config_ref
    loss_cfgs = [loss_cfgs[i] for i in train_view]

    epoch_total_step = 0
    iter_total_step = 0

    ##########################
    # load from checkpoint
    ckpts = [os.path.join(expdir, expname, f)
             for f in sorted(os.listdir(os.path.join(expdir, expname))) if 'tar' in f]
    print('Found ckpts', ckpts)

    if len(args.init_from) > 0:
        ckpt_path = os.path.join(args.prefix, args.init_from)
        assert os.path.exists(ckpt_path), f"Trying to load from {ckpt_path} but it doesn't exist"
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        state_dict = ckpt['network_state_dict']
        nerf.module.init_from_mpi(state_dict)
        nerf.to(device)

    # begin of run one iteration (one patch)
    def run_iter(stepi, optimizer_, datainfo_):
        datainfo_ = [d.to(device) if torch.is_tensor(d) else d for d in datainfo_]
        h_starts, w_starts, b_pose, b_intrin, b_rgbs, loss_cfg = datainfo_
        if args.fp16:
            b_rgbs = b_rgbs.half()
        b_extrin = pose2extrin_torch(b_pose)
        patch_h, patch_w = b_rgbs.shape[-2:]

        if args.add_intrin_noise:
            dxy = torch.rand(2).type_as(b_intrin) - 0.5  # half pixel
            b_intrin = b_intrin.clone()
            b_intrin[:, :2, 2] += dxy

        nerf.train()
        rgb, extra = nerf(patch_h, patch_w, b_extrin, b_intrin, res=b_rgbs, losscfg=loss_cfg)

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

        optimizer_.zero_grad()
        loss.backward()
        optimizer_.step()

        if (stepi + 1) % args.i_img == 0:
            writer.add_scalar('aloss/swd', swd_loss.item(), stepi)
            for k, v in extra.items():
                writer.add_scalar(f'{k}', float(v.mean()), stepi)
            for name, newlr in name_lrates:
                writer.add_scalar(f'lr/{name}', newlr, stepi)

        if (stepi + 1) % args.i_print == 0:
            epoch_tqdm.set_description(f"[TRAIN] Iter: {stepi} Loss: {loss.item():.4f} SWD: {swd_loss.item():.4f}",
                                       "|".join([f"{k}: {v.item():.4f}" for k, v in extra_losses.items()]))

    # end of run one iteration

    # ##########################
    # start training
    # ##########################
    print('Begin')
    for pyr_i, (train_factor, hw, num_epoch) in enumerate(zip(pyr_factors, pyr_hw, pyr_num_epoch)):
        nerf.module.lod(train_factor)
        optimizer = nerf.module.get_optimizer(step=0)
        torch.cuda.empty_cache()
        # generate dataset and optimizer
        dataset = MVVidPatchDataset(hw, videos,
                                    (args.patch_h_size, args.patch_w_size),
                                    (args.patch_h_stride, args.patch_w_stride),
                                    poses, intrins, loss_configs=loss_cfgs)
        dataloader = DataLoader(dataset, 1, shuffle=True)
        epoch_tqdm = trange(num_epoch)
        for epoch_i in epoch_tqdm:
            for iter_i, datainfo in enumerate(dataloader):

                if hasattr(nerf.module, "update_step"):
                    nerf.module.update_step(epoch_total_step)

                # update learning rate
                name_lrates = nerf.module.get_lrate(epoch_i)

                if args.lrate_adaptive:
                    name_lrates = [(n_, lr_ / len(dataset)) for n_, lr_ in name_lrates]

                for (lrname, new_lrate), param_group in zip(name_lrates, optimizer.param_groups):
                    param_group['lr'] = new_lrate

                # train for one interation
                run_iter(iter_total_step, optimizer, datainfo)

                iter_total_step += 1

            # saving after epoch
            if (epoch_total_step + 1) % args.i_weights == 0:
                save_path = os.path.join(expdir, expname, f'l{pyr_i}_epoch_{epoch_i:04d}.tar')
                save_dict = {
                    'epoch_i': epoch_i,
                    'epoch_total_step': epoch_total_step,
                    'iter_total_step': iter_total_step,
                    'pyr_i': pyr_i,
                    'train_factor': train_factor,
                    'hw': hw,
                    'network_state_dict': nerf.module.state_dict(),
                }
                torch.save(save_dict, save_path)

            if (epoch_total_step + 1) % args.i_video == 0:
                moviebase = os.path.join(expdir, expname, f'l{pyr_i}_{epoch_i:04d}_')
                if hasattr(nerf.module, "save_mesh"):
                    prefix = os.path.join(expdir, expname, f"mesh_l{pyr_i}_{epoch_i:04d}")
                    nerf.module.save_mesh(prefix)

                if hasattr(nerf.module, "save_texture"):
                    prefix = os.path.join(expdir, expname, f"texture_l{pyr_i}_{epoch_i:04d}")
                    nerf.module.save_texture(prefix)

                print('render poses shape', render_extrins.shape, render_intrins.shape)
                with torch.no_grad():
                    nerf.eval()

                    rgbs = []
                    for ri in range(len(render_extrins)):
                        r_pose = render_extrins[ri:ri + 1]
                        r_intrin = render_intrins[ri:ri + 1]

                        rgb, extra = nerf(H, W, r_pose, r_intrin, ts=[ri % args.mpv_frm_num])
                        rgb = rgb[0].permute(1, 2, 0).cpu().numpy()
                        rgbs.append(rgb)

                    rgbs = np.array(rgbs)
                    imageio.mimwrite(moviebase + '_rgb.mp4', to8b(rgbs), fps=FPS, quality=8)

            epoch_total_step += 1


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train(args)
