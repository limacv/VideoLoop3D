import os
import torch
import imageio
import numpy as np
import math
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from MPI import *
from torch.utils.data import Dataset, DataLoader
from dataloader import load_mv_videos, poses_avg, load_llff_data
from utils import *
import shutil
from datetime import datetime
from metrics import compute_img_metric
import cv2
from tqdm import tqdm, trange
from config_parser import config_parser


class MVPatchDataset(Dataset):
    def __init__(self, resize_hw, videos, patch_size, patch_stride, poses, intrins, mode='average'):
        super().__init__()
        h_raw, w_raw, _ = videos[0][0].shape[-3:]
        self.h, self.w = resize_hw
        self.v = len(videos)
        self.poses = poses.clone().cpu()
        self.intrins = intrins.clone().cpu()
        self.intrins[:, :2] *= torch.tensor([self.w / w_raw, self.h / h_raw]).reshape(1, 2, 1).type_as(intrins)
        self.patch_h_size, self.patch_w_size = patch_size
        self.mode = mode
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

        self.images = []
        self.dynmask = []
        # for debug only
        # self.images = [torch.rand(3, self.h, self.w)] * self.v
        # self.dynmask = [torch.rand(self.h, self.w)] * self.v
        # return
        for video in videos:
            vid = np.array([cv2.resize(img, (self.w, self.h)) for img in video]) / 255
            # mid
            if self.mode == 'median':
                img = np.median(vid, axis=0)
            elif self.mode == 'average':
                # aveage
                img = vid.mean(axis=0)
            elif self.mode == 'first':
                img = vid[0]
            elif self.mode.startswith('dynamic'):
                # emphsize the dynamics
                weight = np.linalg.norm(vid - vid.mean(axis=0, keepdims=True), axis=-1, keepdims=True)
                k = self.mode.lstrip('dynamic')
                k = 1 if len(k) == 0 else float(k)
                weight = k * weight + (1 - k)
                weight = np.clip(weight, 1e-10, 999999)
                img = (vid * weight).sum(axis=0) / weight.sum(axis=0)
            elif self.mode.startswith('blur'):
                b = self.mode.lstrip('blur')
                b = 11 if len(b) == 0 else int(b)
                vid_blur = np.array([cv2.GaussianBlur(v_, (b, b), 0) for v_ in vid])
                vid_blur_avg = vid_blur.mean(axis=0, keepdims=True)
                weight = np.linalg.norm(vid_blur - vid_blur_avg, axis=-1, keepdims=True)
                weight = np.clip(weight * 3, 0.001, 3)
                img = (vid_blur * weight).sum(axis=0) / weight.sum(axis=0)
            else:
                raise RuntimeError(f"Unrecognized vid2img_mode={self.mode}")

            img = torch.tensor(img).permute(2, 0, 1)
            loopmask = compute_loopable_mask(vid)
            loopmask = torch.tensor(loopmask).type_as(img)
            self.images.append(img)
            self.dynmask.append(loopmask)
        print(f"Dataset: generate {len(self)} patches for training, pad {pad_info} to videos")

    def __len__(self):
        return len(self.patch_wh_start)

    def __getitem__(self, item):
        w_start, h_start = self.patch_wh_start[item]
        view_idx = self.view_index[item]
        pose = self.poses[view_idx]
        intrin = get_new_intrin(self.intrins[view_idx], h_start, w_start).float()
        crops = self.images[view_idx][..., h_start: h_start + self.patch_h_size, w_start: w_start + self.patch_w_size]
        crops_ma = self.dynmask[view_idx][h_start: h_start + self.patch_h_size, w_start: w_start + self.patch_w_size]
        return w_start, h_start, pose, intrin, crops.cuda(), crops_ma.cuda()


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
    datadir = os.path.join(args.prefix, args.datadir)
    expdir = os.path.join(args.prefix, args.expdir)
    videos, _, poses, intrins, bds, render_poses, render_intrins = load_mv_videos(basedir=datadir,
                                                                                  factor=args.factor,
                                                                                  bd_factor=args.bd_factor,
                                                                                  recenter=True)
    H, W = videos[0][0].shape[0:2]
    V = len(videos)
    print('Loaded llff', V, H, W, poses.shape, intrins.shape, render_poses.shape, bds.shape)

    ref_pose = poses_avg(poses)[:, :4]
    ref_extrin = pose2extrin_np(ref_pose)
    ref_intrin = intrins.mean(0)
    ref_near, ref_far = bds[:, 0].min(), bds[:, 1].max()

    # Summary writers
    writer = SummaryWriter(os.path.join(expdir, args.expname))

    # Create log dir and copy the config file
    if not args.render_only:
        file_path = os.path.join(expdir, args.expname, f"source_{datetime.now().timestamp():.0f}")
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
    optimizer = nerf.module.get_optimizer()

    render_extrins = pose2extrin_np(render_poses)
    render_extrins = torch.tensor(render_extrins).float()
    render_intrins = torch.tensor(render_intrins).float()

    ######################
    # if optimize poses
    poses = torch.tensor(poses)
    intrins = torch.tensor(intrins)

    ##########################
    # Load checkpoints
    ckpts = [os.path.join(expdir, args.expname, f)
             for f in sorted(os.listdir(os.path.join(expdir, args.expname))) if 'tar' in f]
    print('Found ckpts', ckpts)

    start = 0
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['epoch_i']
        nerf.module.load_state_dict(ckpt['network_state_dict'])

    # begin of run one iteration (one patch)
    def run_iter(stepi, optimizer_, datainfo_):
        h_starts, w_starts, b_pose, b_intrin, b_rgbs, b_loopmask = datainfo_
        b_extrin = pose2extrin_torch(b_pose)
        patch_h, patch_w = b_rgbs.shape[-2:]

        nerf.train()
        rgb, extra = nerf(patch_h, patch_w, b_extrin, b_intrin)
        if args.learn_loop_mask:
            loop_mask = rgb[:, -1]
            loop_loss = img2mse(loop_mask, b_loopmask)
            rgb = rgb[:, :3]
        else:
            loop_loss = 0

        # RGB loss
        img_loss = img2mse(rgb, b_rgbs)
        psnr = mse2psnr(img_loss)

        # define extra losses here
        args_var = vars(args)
        extra_losses = {}
        for k, v in extra.items():
            if args_var[f"{k}_loss_weight"] > 0:
                extra_losses[k] = extra[k].mean() * args_var[f"{k}_loss_weight"]

        loss = img_loss + loop_loss
        for v in extra_losses.values():
            loss = loss + v

        optimizer_.zero_grad()
        loss.backward()
        if hasattr(nerf.module, "post_backward"):
            nerf.module.post_backward()
        optimizer_.step()

        if stepi % args.i_img == 0:
            writer.add_scalar('aloss/psnr', psnr, stepi)
            writer.add_scalar('aloss/mse_loss', loss, stepi)
            for k, v in extra.items():
                writer.add_scalar(f'{k}', float(v.mean()), stepi)
            for name, newlr in name_lrates:
                writer.add_scalar(f'lr/{name}', newlr, stepi)

        if stepi % args.i_print == 0:
            epoch_tqdm.set_description(f"[TRAIN] Iter: {stepi} Loss: {loss.item():.4f} PSNR: {psnr.item():.4f}",
                                       "|".join([f"{k}: {v.item():.4f}" for k, v in extra_losses.items()]))

    # end of run one iteration

    # ##########################
    # start training
    # ##########################
    print('Begin')

    dataset = MVPatchDataset((H, W), videos,
                             (args.patch_h_size, args.patch_w_size),
                             (args.patch_h_stride, args.patch_w_stride),
                             poses, intrins, args.vid2img_mode)
    # visualize the image, delete afterwards
    for viewi, (img, loopma) in enumerate(zip(dataset.images, dataset.dynmask)):
        p = os.path.join(expdir, args.expname, f"imgvis_{args.vid2img_mode}", f"{viewi:04d}.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        imageio.imwrite(p, to8b(img.permute(1, 2, 0).cpu().numpy()))
        pm = os.path.join(expdir, args.expname, f"loopvis", f"{viewi:04d}.png")
        os.makedirs(os.path.dirname(pm), exist_ok=True)
        imageio.imwrite(pm, to8b(loopma.cpu().numpy()))
    dataloader = DataLoader(dataset, 1, shuffle=True)

    iter_total_step = 0
    epoch_tqdm = trange(args.N_iters)
    for epoch_i in epoch_tqdm:
        if epoch_i < start:
            continue

        if epoch_i == args.sparsify_epoch:
            print("Sparsifying mesh models")
            nerf.module.sparsify_faces(alpha_thresh=args.sparsify_alpha_thresh)
            optimizer = nerf.module.get_optimizer()

        if epoch_i == args.direct2sh_epoch:
            print("Converting direct to data_sh")
            nerf.module.direct2sh()
            optimizer = nerf.module.get_optimizer()

        for iter_i, datainfo in enumerate(dataloader):
            if hasattr(nerf.module, "update_step"):
                nerf.module.update_step(iter_total_step)

            ###   update learning rate   ###
            name_lrates = nerf.module.get_lrate(iter_total_step)
            for (lrname, new_lrate), param_group in zip(name_lrates, optimizer.param_groups):
                param_group['lr'] = new_lrate

            # train for one interation
            datainfo = [d.cuda() for d in datainfo]
            run_iter(iter_total_step, optimizer, datainfo)

            iter_total_step += 1

        ################################
        if (epoch_i + 1) % args.i_weights == 0:
            save_path = os.path.join(expdir, args.expname, f'epoch_{epoch_i:04d}.tar')
            save_dict = {
                'epoch_i': epoch_i,
                'network_state_dict': nerf.module.state_dict()
            }
            torch.save(save_dict, save_path)

        if (epoch_i + 1) % args.i_video == 0:
            moviebase = os.path.join(expdir, args.expname, f'epoch_{epoch_i:04d}_')
            if hasattr(nerf.module, "save_mesh"):
                prefix = os.path.join(expdir, args.expname, f"mesh_epoch_{epoch_i:04d}")
                nerf.module.save_mesh(prefix)

            if hasattr(nerf.module, "save_texture"):
                prefix = os.path.join(expdir, args.expname, f"texture_epoch_{epoch_i:04d}")
                nerf.module.save_texture(prefix)

            if args.learn_loop_mask and hasattr(nerf.module, "save_loopmask"):
                prefix = os.path.join(expdir, args.expname, f"loopable_epoch_{epoch_i:04d}")
                nerf.module.save_loopmask(prefix)

            print('render poses shape', render_extrins.shape, render_intrins.shape)
            with torch.no_grad():
                nerf.eval()

                rgbs = []
                loopmasks = []
                for ri in range(len(render_extrins)):
                    r_pose = render_extrins[ri:ri + 1]
                    r_intrin = render_intrins[ri:ri + 1]

                    rgbl, extra = nerf(H, W, r_pose, r_intrin)
                    if args.learn_loop_mask:
                        rgb, loopmask = rgbl[:, :3], rgbl[:, -1]
                        loopmask = loopmask[0].cpu().numpy()
                        loopmask = np.stack([np.zeros_like(loopmask), loopmask, np.zeros_like(loopmask)], -1)
                        loopmasks.append(loopmask)
                    else:
                        rgb = rgbl
                    rgb = rgb[0].permute(1, 2, 0).cpu().numpy()
                    rgbs.append(rgb)

                rgbs = np.array(rgbs)
                imageio.mimwrite(moviebase + '_rgb.mp4', to8b(rgbs), fps=30, quality=8)
                if len(loopmasks) > 0:
                    loopmasks = np.array(loopmasks)
                    imageio.mimwrite(moviebase + '_loopable.mp4', to8b(loopmasks), fps=30, quality=8)


if __name__ == '__main__':
    train()
