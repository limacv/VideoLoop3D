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
from metrics import compute_img_metric
import cv2
from config_parser import config_parser
from tqdm import tqdm, trange


class MVVidPatchDataset(Dataset):
    def __init__(self, resize_hw, videos, patch_size, patch_stride, poses, intrins):
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

        return w_start, h_start, pose, intrin, crops.cuda()


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
    videos, poses, intrins, bds, render_poses, render_intrins = load_mv_videos(basedir=args.datadir,
                                                                               factor=args.factor,
                                                                               bd_factor=args.bd_factor,
                                                                               recenter=True)

    H, W = videos[0][0].shape[0:2]
    V = len(videos)
    print('Loaded llff', V, H, W, poses.shape, intrins.shape, render_poses.shape, bds.shape)

    ref_pose = poses_avg(poses)[:, :4]
    ref_extrin = pose2extrin_np(ref_pose)
    ref_intrin = intrins[0]
    ref_near, ref_far = bds[:, 0].min(), bds[:, 1].max()

    # Resove pyramid related configs, controled by (pyr_stage, pyr_factor, N_iters)
    #                                           or (pyr_minimal_dim, pyr_factor, pyr_num_step)
    if args.pyr_minimal_dim < 0:
        # store the iter_num when starting the stage
        pyr_stages = list(map(int, args.pyr_stage.split(','))) if len(args.pyr_stage) > 0 else []
        pyr_stages = np.array([0] + pyr_stages + [args.N_iters])  # one default stage
        pyr_num_step = pyr_stages[1:] - pyr_stages[:-1]
        pyr_factors = [args.pyr_factor ** i for i in list(range(len(pyr_num_step)))[::-1]]
        pyr_hw = [(int(H * f), int(W * f)) for f in pyr_factors]
    else:
        num_stage = int(np.log(args.pyr_minimal_dim / min(H, W)) / np.log(args.pyr_factor)) + 1
        pyr_factors = [args.pyr_factor ** i for i in list(range(num_stage))[::-1]]
        pyr_hw = [(int(H * f), int(W * f)) for f in pyr_factors]
        pyr_num_step = [args.pyr_num_step] * num_stage
    print("Pyramid info: ")
    for leveli, (f_, hw_, num_step_) in enumerate(zip(pyr_factors, pyr_hw, pyr_num_step)):
        print(f"    level {leveli}: factor {f_} [{hw_[0]} x {hw_[1]}] run for {num_step_} iterations")
    # end of pyramid infomation

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
    if args.model_type == "MPMesh":
        nerf = MPMeshVid(args, H, W, ref_extrin, ref_intrin, ref_near, ref_far)
    else:
        raise RuntimeError(f"Unrecognized model type {args.model_type}")

    nerf = nn.DataParallel(nerf, list(range(args.gpu_num)))
    nerf.cuda()
    render_extrins = pose2extrin_np(render_poses)
    render_extrins = torch.tensor(render_extrins).float()
    render_intrins = torch.tensor(render_intrins).float()

    poses = torch.tensor(poses)
    intrins = torch.tensor(intrins)

    ##########################
    # initialize the MPV
    # ckpts = [os.path.join(args.expdir, args.expname, f)
    #          for f in sorted(os.listdir(os.path.join(args.expdir, args.expname))) if 'tar' in f]
    # print('Found ckpts', ckpts)
    #
    # start = 0
    # if len(ckpts) > 0 and not args.no_reload:
    #     ckpt_path = ckpts[-1]
    #     print('Reloading from', ckpt_path)
    #     ckpt = torch.load(ckpt_path)
    #
    #     start = ckpt['global_step']
    #     optimizer = nerf.module.get_optimizer(start)
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #     smart_load_state_dict(nerf, ckpt)

    # begin of run one iteration (one patch)
    def run_iter(stepi, optimizer_, datainfo_):
        h_starts, w_starts, b_pose, b_intrin, b_rgbs = datainfo_
        b_extrin = pose2extrin_torch(b_pose)
        patch_h, patch_w = b_rgbs.shape[-2:]

        nerf.train()
        rgb, extra = nerf(patch_h, patch_w, b_extrin, b_intrin, res=b_rgbs)

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

        if stepi % args.i_img == 0:
            writer.add_scalar('aloss/swd', swd_loss.item(), stepi)
            for k, v in extra.items():
                writer.add_scalar(f'{k}', float(v.mean()), stepi)
            for name, newlr in name_lrates:
                writer.add_scalar(f'lr/{name}', newlr, stepi)

        if stepi % args.i_print == 0:
            print(f"[TRAIN] Iter: {stepi} Loss: {loss.item():.4f} SWD: {swd_loss.item():.4f}",
                  "|".join([f"{k}: {v.item():.4f}" for k, v in extra_losses.items()]))
    # end of run one iteration

    # ##########################
    # start training
    # ##########################
    epoch_total_step = 0
    iter_total_step = 0
    # TODO: figure out pyramid level based on the start
    print('Begin')
    for pyr_i, (train_factor, hw, num_step) in enumerate(zip(pyr_factors, pyr_hw, pyr_num_step)):
        nerf.module.lod(train_factor)
        optimizer = nerf.module.get_optimizer(step=0)
        # generate dataset and optimizer
        dataset = MVVidPatchDataset(hw, videos,
                                    (args.patch_h_size, args.patch_w_size),
                                    (args.patch_h_stride, args.patch_w_stride),
                                    poses, intrins)
        dataloader = DataLoader(dataset, 1, shuffle=True)
        for epoch_i in trange(num_step):
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
                datainfo = [d.cuda() for d in datainfo]
                run_iter(iter_total_step, optimizer, datainfo)

                iter_total_step += 1

            # saving after epoch
            if epoch_total_step % args.i_weights == 0:
                if hasattr(nerf.module, "save_mesh"):
                    prefix = os.path.join(args.expdir, args.expname, f"mesh_l{pyr_i}_{epoch_i:06d}")
                    nerf.module.save_mesh(prefix)

                if hasattr(nerf.module, "save_texture"):
                    prefix = os.path.join(args.expdir, args.expname, f"texture_l{pyr_i}_{epoch_i:06d}")
                    nerf.module.save_texture(prefix)

            if epoch_total_step % args.i_video == 0:
                moviebase = os.path.join(args.expdir, args.expname, f'l{pyr_i}_{epoch_i:06d}_')
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
                    imageio.mimwrite(moviebase + '_rgb.mp4', to8b(rgbs), fps=30, quality=8)

            epoch_total_step += 1


if __name__ == '__main__':
    train()
