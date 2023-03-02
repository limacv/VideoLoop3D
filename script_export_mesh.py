import torch
import os
import numpy as np
import imageio
import json
from dataloader import load_llff_data
from utils import save_obj_with_vcolor, save_obj_multimaterial, normalize_uv, cull_unused


def export_mpv_repr(cfg_file, cfg_file1):
    from config_parser import config_parser
    parser = config_parser()
    args = parser.parse_args(["--config", cfg_file, "--config1", cfg_file1])

    prefix = args.prefix
    expname = args.expname + args.expname_postfix
    outpath = os.path.join(prefix, "mesh4demo", expname)
    os.makedirs(outpath, exist_ok=True)

    data_dir = os.path.join(prefix, args.datadir)
    _, poses, intrins, bds, render_poses, render_intrins = \
        load_llff_data(data_dir, args.factor, False,
                       bd_factor=(args.near_factor, args.far_factor),
                       load_img=False)

    # figuring out the camera geometry
    normalize = lambda x: x / np.linalg.norm(x)
    up = normalize(poses[:, :3, 1].sum(0)).tolist()
    up[1] = -up[1]

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    focal = 1. / (((1. - .75) / close_depth + .75 / inf_depth))
    # Get radii for spiral path
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.abs(tt).max(0) * 0.8
    f, cy = intrins[:, 0, 0].mean(), intrins[:, 1, -1].mean()

    json_dict = {
        "fps": 25,
        "fov": np.arctan(cy / f) * 2 / np.pi * 180,
        "frame_count": args.mpv_frm_num,
        "near": float(bds.min()),
        "far": float(bds.max()),

        "up": up,
        "lookat": [0, 0, focal],
        "limit": rads.tolist(),
    }
    jsonobj = json.dumps(json_dict, indent=4)
    with open(os.path.join(outpath, "meta.json"), 'w') as f:
        f.write(jsonobj)

    # saving others
    ckpt_file = os.path.join(prefix, args.expdir, expname, "l5_epoch_0049.tar")
    state_dict = torch.load(ckpt_file)
    state_dict = state_dict['network_state_dict']

    atlas_h_static, atlas_w_static = state_dict["self.atlas_full_h"], state_dict["self.atlas_full_w"]
    atlas_h_dynamic, atlas_w_dynamic = state_dict["self.atlas_full_dyn_h"], state_dict["self.atlas_full_dyn_w"]

    verts = state_dict['_verts'].cpu().numpy()

    # static mesh
    uvs_static = state_dict['uvs'].cpu().numpy()
    faces_static = state_dict['faces'].cpu().numpy()
    uvfaces_static = state_dict['uvfaces'].cpu().numpy()
    atlas_static = torch.sigmoid(state_dict['atlas'])
    atlas_static = np.clip(atlas_static[0].permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)

    # dynamic mesh
    uvs_dynamic = state_dict['uvs_dyn'].cpu().numpy()
    faces_dynamic = state_dict['faces_dyn'].cpu().numpy()
    uvfaces_dynamic = state_dict['uvfaces_dyn'].cpu().numpy()
    atlas_dynamic = torch.sigmoid(state_dict['atlas_dyn'])
    atlas_dynamic = np.clip(atlas_dynamic.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    frame_num = len(atlas_dynamic)
    assert frame_num == args.mpv_frm_num, "Error: detect unmatched frame count"
    # saving geometry

    uvs_static = normalize_uv(uvs_static, atlas_h_static, atlas_w_static)
    uvs_dynamic = normalize_uv(uvs_dynamic, atlas_h_dynamic, atlas_w_dynamic)

    # save static
    staticv, staticf = cull_unused(verts, faces_static)
    staticuv, staticuvf = cull_unused(uvs_static, uvfaces_static)
    staticcolor = np.zeros_like(staticv)
    staticcolor[:, 0] = 1
    staticvc = np.concatenate([staticv, staticcolor], -1)

    dynamicv, dynamicf = cull_unused(verts, faces_dynamic)
    dynamicuv, dynamicuvf = cull_unused(uvs_dynamic, uvfaces_dynamic)
    dynamiccolor = np.zeros_like(dynamicv)
    dynamiccolor[:, 1] = 1
    dynamicvc = np.concatenate([dynamicv, dynamiccolor], -1)

    # concate two meshes
    newv = np.concatenate([staticvc, dynamicvc])
    newuv = np.concatenate([staticuv, dynamicuv])
    newf = np.concatenate([staticf, dynamicf + len(staticvc)])
    newuvf = np.concatenate([staticuvf, dynamicuvf + len(staticuv)])

    # order the face
    depth = newv[newf[:, 0]][:, 2]
    ordr = np.argsort(depth)[::-1]
    newf = newf[ordr]
    newuvf = newuvf[ordr]

    save_obj_with_vcolor(os.path.join(outpath, "geometry.obj"),
                         newv, newf, newuv, newuvf)

    imageio.imwrite(os.path.join(outpath, "static.png"), atlas_static)
    vidoutpath = os.path.join(outpath, "dynamic")
    os.makedirs(vidoutpath, exist_ok=True)
    for i in range(frame_num):
        imageio.imwrite(os.path.join(vidoutpath, f"{i:04d}.png"),
                        atlas_dynamic[i])


if __name__ == "__main__":
    cfg1s = [
        # "108fall1",
        # "108fall2",
        # "108fall3",
        "108fall4",
        # "108fall5",
        # "110grasstree",
        # "110pillar",
        # "1017palm",
        # "1017yuan",
        # "1020rock",
        # "1020ustfall1",
        # "1020ustfall2",
        # "1101grass",
        # "1101towerd",
        # "ustfallclose",
        # "usttap"
    ]

    for cfg1 in cfg1s:
        export_mpv_repr(
            "configs/mpv_base.txt",
            f"configs/mpvs/{cfg1}.txt",
        )


