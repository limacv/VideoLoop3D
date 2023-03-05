import torch
import os
import numpy as np
import imageio
import json
from dataloader import load_llff_data
from config_parser import config_parser
from utils import save_obj_with_vcolor, save_obj_multimaterial, normalize_uv, cull_unused

PATCH_SIZE = 16


# merge the neighbor pixels to prevent tiling artifact
# failed experiment, turns out this will leads to artifact
# so this function is currently unused
def merge_neighbor_pixels(v, f, uv, uvf, atlas):  # len(f) == len(uvf)
    # faces of a quad: 0 - 1   two face is arraged as (0, 1, 3), (3, 2, 0)
    #                  | \ |
    #                  2 - 3
    h, w, _ = atlas.shape
    f = f.reshape(-1, 6)
    uvf = uvf.reshape(-1, 6)

    # find neighbor edge (horizon
    edge_h = f[:, [0, 4, 1, 2]].reshape(-1, 2)
    edge_h_uv = uvf[:, [0, 4, 1, 2]].reshape(-1, 2)
    edge_h_flat = edge_h[:, 0] + edge_h[:, 1] * len(v)
    sortidx = np.argsort(edge_h_flat)
    edge_h_flat = edge_h_flat[sortidx]
    edge_h_uv = edge_h_uv[sortidx]
    _, idx, counts = np.unique(edge_h_flat, return_index=True, return_counts=True)
    idx0 = idx[np.argwhere(counts == 2)][:, 0]
    idx1 = idx0 + 1

    uv_select = uv[edge_h_uv[idx0]]
    uv_select1 = uv[edge_h_uv[idx1]]
    # | |
    x_idx0, x_idx1, y_idx0, y_idx1 = uv_select[:, 0, 0], uv_select1[:, 0, 0], uv_select[:, 0, 1], uv_select1[:, 0, 1]
    x_idx0 = np.round((x_idx0 + 1) / (2 / (w - 1))).astype(np.int32)
    x_idx1 = np.round((x_idx1 + 1) / (2 / (w - 1))).astype(np.int32)
    y_idx0 = np.round((y_idx0 + 1) / (2 / (h - 1))).astype(np.int32)
    y_idx1 = np.round((y_idx1 + 1) / (2 / (h - 1))).astype(np.int32)
    rang = np.arange(PATCH_SIZE)
    pix_loc0 = [np.stack([x_idx0[:, None].repeat(len(rang), 1), y_idx0[:, None] + rang[None]], axis=-1)]
    pix_loc1 = [np.stack([x_idx1[:, None].repeat(len(rang), 1), y_idx1[:, None] + rang[None]], axis=-1)]

    # find neighbor edge (vertical)
    edge_v = f[:, [0, 1, 4, 2]].reshape(-1, 2)
    edge_v_uv = uvf[:, [0, 1, 4, 2]].reshape(-1, 2)
    edge_v_flat = edge_v[:, 0] + edge_v[:, 1] * len(v)
    sortidx = np.argsort(edge_v_flat)
    edge_v_flat = edge_v_flat[sortidx]
    edge_v_uv = edge_v_uv[sortidx]
    _, idx, counts = np.unique(edge_v_flat, return_index=True, return_counts=True)
    idx0 = idx[np.argwhere(counts == 2)][:, 0]
    idx1 = idx0 + 1

    uv_select = uv[edge_v_uv[idx0]]
    uv_select1 = uv[edge_v_uv[idx1]]
    # --
    # --
    x_idx0, x_idx1, y_idx0, y_idx1 = uv_select[:, 0, 0], uv_select1[:, 0, 0], uv_select[:, 0, 1], uv_select1[:, 0, 1]
    x_idx0 = np.round((x_idx0 + 1) / (2 / (w - 1))).astype(np.int32)
    x_idx1 = np.round((x_idx1 + 1) / (2 / (w - 1))).astype(np.int32)
    y_idx0 = np.round((y_idx0 + 1) / (2 / (h - 1))).astype(np.int32)
    y_idx1 = np.round((y_idx1 + 1) / (2 / (h - 1))).astype(np.int32)
    rang = np.arange(PATCH_SIZE)
    pix_loc0 += [np.stack([x_idx0[:, None] + rang[None], y_idx0[:, None].repeat(len(rang), 1)], axis=-1)]
    pix_loc1 += [np.stack([x_idx1[:, None] + rang[None], y_idx1[:, None].repeat(len(rang), 1)], axis=-1)]

    pix1 = np.concatenate(pix_loc0).reshape(-1, 2)
    pix2 = np.concatenate(pix_loc1).reshape(-1, 2)
    return pix1, pix2


def export_mpv_repr(args):
    prefix = args.prefix
    expname = args.expname + args.expname_postfix
    outpath = os.path.join(prefix, args.mesh_folder, expname)
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

    # pix1, pix2 = merge_neighbor_pixels(verts, faces_static, uvs_static, uvfaces_static, atlas_static)
    # color1, color2 = atlas_static[pix1[:, 1], pix1[:, 0]], atlas_static[pix2[:, 1], pix2[:, 0]]
    # color = np.minimum(color1, color2)
    # atlas_static[pix1[:, 1], pix1[:, 0]] = color
    # atlas_static[pix2[:, 1], pix2[:, 0]] = color
    # # will chagne atlas_dynamic
    # pix1, pix2 = merge_neighbor_pixels(verts, faces_dynamic, uvs_dynamic, uvfaces_dynamic, atlas_dynamic[0])
    # color1, color2 = atlas_dynamic[:, pix1[:, 1], pix1[:, 0]], atlas_dynamic[:, pix2[:, 1], pix2[:, 0]]
    # color = np.minimum(color1, color2)
    # atlas_dynamic[:, pix1[:, 1], pix1[:, 0]] = color
    # atlas_dynamic[:, pix2[:, 1], pix2[:, 0]] = color

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
    parser = config_parser()
    parser.add_argument("--mesh_folder", type=str, default="meshes",
                        help='')
    args = parser.parse_args()
    export_mpv_repr(args)
