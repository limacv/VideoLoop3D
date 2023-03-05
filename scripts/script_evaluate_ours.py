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
from evaluations.SVFID import svfid
from evaluations.LPIPS import compute_lpips, compute_lpips_slidewindow
from evaluations.NNMSE import compute_nnerr

# Flag
COMPUTE_STATIC = True
COMPUTE_DYN = True
COMPUTE_LPIPS = True
COMPUTE_NNMSE = True
COMPUTE_LOOPQ = True
COMPUTE_SVFID = True


def evaluate(args):
    device = 'cuda:0'
    if args.gpu_num <= 0:
        device = 'cpu'
        print(f"Using CPU for training")

    expname = args.expname + args.expname_postfix
    print(f"Evaluating: {expname}")
    args.datadir = args.datadir.rstrip('/\\')
    if args.datadir.endswith("_loop"):
        print(f"Warning!!! Detect data pointing to the looping dataset, "
              f"will change from {args.datadir} to {args.datadir[:-5]}")
        args.datadir = args.datadir[:-5]

    datadir = os.path.join(args.prefix, args.datadir)
    expdir = os.path.join(args.prefix, args.expdir)
    videos, FPS, poses, intrins, bds, render_poses, render_intrins = \
        load_mv_videos(basedir=datadir,
                       factor=args.factor,
                       bd_factor=(args.near_factor, args.far_factor),
                       recenter=True)

    H, W = videos[0][0].shape[0:2]
    print('Loaded llff', H, W, poses.shape, intrins.shape, render_poses.shape, bds.shape)
    test_view = args.test_view_idx
    test_view = list(map(int, test_view.split(','))) if len(test_view) > 0 else list(range(V))
    # filter out test view
    videos = [videos[train_i] for train_i in test_view]
    videos = [np.array(vid) for vid in videos]
    poses = poses[test_view]
    intrins = intrins[test_view]
    print(f'Test view: {test_view}')
    V = len(videos)

    # generate loopmask
    loopmasks = [compute_loopable_mask(v_ / 255) for v_ in videos]
    loopmasks = [- np.array(m_).astype(np.float32) + 1 for m_ in loopmasks]
    ref_pose = poses_avg(poses)[:, :4]
    ref_extrin = pose2extrin_np(ref_pose)
    ref_intrin = intrins[0]
    ref_near, ref_far = bds.min(), bds.max()

    # Create nerf model
    if args.model_type == "MPMesh":
        args.mpi_h_scale = args.mpi_w_scale = 0.01
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
    # evaluating ours
    # ##########################
    ours_rgb = []
    print('Begin')
    moviebase = os.path.join(expdir, expname, f'eval_')
    with torch.no_grad():
        nerf.eval()

        for viewi in range(V):
            torch.cuda.empty_cache()
            r_pose = extrins[viewi: viewi + 1]
            r_intrin = intrins[viewi: viewi + 1]
            ts = torch.arange(nerf.module.frm_num).long()
            rgb = [nerf(H, W, r_pose, r_intrin, ts[ti: ti + 2])[0] for ti in range(0, len(ts), 2)]
            rgb = torch.concat(rgb)
            rgb = rgb.permute(0, 2, 3, 1).cpu().numpy()
            rgb = to8b(rgb)
            ours_rgb.append(rgb)

        # ########################
        # Computing metrics. gt, pred are videos F x H x W x 3, in (0, 255), rgb
        # ########################
        crop = 40
        videos = [vid[:, crop:-crop, crop:-crop] for vid in videos]
        ours_rgb = [vid[:, crop:-crop, crop:-crop] for vid in ours_rgb]
        loopmasks = [m_[crop:-crop, crop:-crop] for m_ in loopmasks]
        # torch.cuda.empty_cache()
        # fids = []
        # print("computing svfid error")
        # for viewi in trange(V):
        #     gt = videos[viewi]
        #     pred = ours_rgb[viewi]
        #     gt = [cv2.resize(gt_[12:12 + 336, 152: 152 + 336], (112, 112)) for gt_ in gt]
        #     pred = [cv2.resize(p_[12:12 + 336, 152: 152 + 336], (112, 112)) for p_ in pred]
        #     gt = torch.tensor(np.array(gt)).cuda().float() / 255
        #     pred = torch.tensor(np.array(pred)).cuda().float() / 255
        #     try:
        #         fid = svfid(gt, pred)
        #     except Exception as e:
        #         print(e)
        #         fid = -1
        #
        #     fids.append(fid)
        if COMPUTE_STATIC:
            torch.cuda.empty_cache()
            print("computing static error")
            static_psnr = []
            static_ssim = []
            from evaluations.metrics import compute_img_metric
            for viewi in trange(V):
                gt = videos[viewi]
                pred = ours_rgb[viewi]
                frm_min = min(len(gt), len(pred))
                gt, pred = gt[:frm_min] / 255, pred[:frm_min] / 255
                lmask = loopmasks[viewi]
                psnr = compute_img_metric(torch.tensor(gt), torch.tensor(pred), "psnr", torch.tensor(lmask[None]))
                ssim = compute_img_metric(torch.tensor(gt), torch.tensor(pred), "ssim", torch.tensor(lmask[None]))
                static_psnr.append(psnr)
                static_ssim.append(ssim)
        else:
            static_psnr = [0] * V
            static_ssim = [1] * V

        if COMPUTE_DYN:
            torch.cuda.empty_cache()
            dyns = []
            print("computing dynamic error")
            for viewi in trange(V):
                gt = videos[viewi]
                pred = ours_rgb[viewi]
                stdgt = np.std(gt, axis=0)
                stdpred = np.std(pred, axis=0)
                err = ((stdgt - stdpred) ** 2).mean()
                dyns.append(err)
        else:
            dyns = [0] * V

        if COMPUTE_LPIPS:
            torch.cuda.empty_cache()
            lpips = []
            lpips_sw = []
            print("computing lpips error")
            for viewi in trange(V):
                gt = videos[viewi]
                pred = ours_rgb[viewi]
                gt = torch.tensor(np.array(gt)).cuda().float()
                pred = torch.tensor(np.array(pred)).cuda().float()
                lpip = compute_lpips(pred, gt)
                lpipsw = compute_lpips_slidewindow(pred, gt)
                lpips.append(lpip)
                lpips_sw.append(lpipsw)
        else:
            lpips = [0] * V
            lpips_sw = [0] * V

        patch_sizes = [5, 11, 17]
        stride_sizes = [2, 4, 6]
        patcht_sizes = [7, 5, 3]
        stridet_sizes = [1, 1, 1]
        if COMPUTE_LOOPQ:
            torch.cuda.empty_cache()
            loop_qualitys = []
            print("computing Loop Quality")
            for viewi in trange(V):
                gt = videos[viewi]
                pred = ours_rgb[viewi]
                gt = torch.tensor(np.array(gt)).cuda().float().permute(3, 0, 1, 2)[None]
                pred = torch.tensor(np.array(pred)).cuda().float().permute(3, 0, 1, 2)[None]

                loop_quality = []
                for i, (psz, ssz, pszt, sszt) in enumerate(zip(patch_sizes, stride_sizes, patcht_sizes, stridet_sizes)):
                    pred_seam = torch.cat([
                        pred[:, :, -pszt + 1:], pred[:, :, :pszt - 1]
                    ], dim=2)
                    loop_quality.append(compute_nnerr(pred_seam, gt, psz, ssz, pszt, sszt))

                loop_qualitys.append(loop_quality)
        else:
            loop_qualitys = [[0] * len(patch_sizes)] * V

        if COMPUTE_NNMSE:
            torch.cuda.empty_cache()
            nnmses_complete = []
            nnmses_coherent = []
            print("computing NN error")
            for viewi in trange(V):
                gt = videos[viewi]
                pred = ours_rgb[viewi]
                gt = torch.tensor(np.array(gt)).cuda().float().permute(3, 0, 1, 2)[None]
                pred = torch.tensor(np.array(pred)).cuda().float().permute(3, 0, 1, 2)[None]

                complete, coherent = [], []
                for i, (psz, ssz, pszt, sszt) in enumerate(zip(patch_sizes, stride_sizes, patcht_sizes, stridet_sizes)):
                    complete.append(compute_nnerr(gt, pred, psz, ssz, pszt, sszt))
                    coherent.append(compute_nnerr(pred, gt, psz, ssz, pszt, sszt))

                nnmses_complete.append(complete)  # forward
                nnmses_coherent.append(coherent)  # backward
        else:
            nnmses_complete = [[0] * len(patch_sizes)] * V
            nnmses_coherent = [[0] * len(patch_sizes)] * V

        mean = lambda x: sum(x) / len(x)
        names = ["name", "nnf", "nnb", "dyn", "lpips", "lpips_sw", "loop", "psnr", "ssim"] + \
                [f"nnf_p{p}s{s}pt{pt}st{st}" for p, s, pt, st in zip(patch_sizes, stride_sizes, patcht_sizes, stridet_sizes)] + \
                [f"nnb_p{p}s{s}pt{pt}st{st}" for p, s, pt, st in zip(patch_sizes, stride_sizes, patcht_sizes, stridet_sizes)] + \
                [f"loop_p{p}s{s}pt{pt}st{st}" for p, s, pt, st in zip(patch_sizes, stride_sizes, patcht_sizes, stridet_sizes)]
        with open(moviebase + "metrics.txt", 'w') as f:
            f.write(", ".join(names) + "\n")
            dataname = os.path.basename(datadir)

            forwards = np.zeros(len(patch_sizes) + 1)
            backwards = np.zeros(len(patch_sizes) + 1)
            loops = np.zeros(len(patch_sizes) + 1)
            for viewi in range(V):
                f.write(f"{dataname}_view{viewi}, ")
                f.write(", ".join(map(str,
                                      [mean(nnmses_complete[viewi]), mean(nnmses_coherent[viewi]),
                                       dyns[viewi], lpips[viewi], lpips_sw[viewi], mean(loop_qualitys[viewi]),
                                       static_psnr[viewi], static_ssim[viewi]])))
                f.write(", ")
                f.write(", ".join(map(str, nnmses_complete[viewi])))
                f.write(", ")
                f.write(", ".join(map(str, nnmses_coherent[viewi])))
                f.write(", ")
                f.write(", ".join(map(str, loop_qualitys[viewi])))
                f.write("\n")

                forwards[:len(patch_sizes)] += nnmses_complete[viewi]
                forwards[-1] += mean(nnmses_complete[viewi])
                backwards[:len(patch_sizes)] += nnmses_coherent[viewi]
                backwards[-1] += mean(nnmses_coherent[viewi])
                loops[:len(patch_sizes)] += loop_qualitys[viewi]
                loops[-1] += mean(loop_qualitys[viewi])

            forwards = forwards / V
            backwards = backwards / V
            loops = loops / V
            f.write(f"{dataname}, ")
            f.write(", ".join(map(str,
                                  [forwards[-1], backwards[-1],
                                   mean(dyns), mean(lpips), mean(lpips_sw), loops[-1],
                                   mean(static_psnr), mean(static_ssim)])))
            f.write(", ")
            f.write(", ".join(map(str, forwards[:-1].tolist())))
            f.write(", ")
            f.write(", ".join(map(str, backwards[:-1].tolist())))
            f.write(", ")
            f.write(", ".join(map(str, loops[:-1].tolist())))
            f.write("\n")


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    evaluate(args)

