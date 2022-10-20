import torch
import os
import numpy as np
import imageio
import json
from dataloader import load_llff_data


def save_obj_multimaterial(file, vertices, faces_list, uvs, uvfaces_list, mtls_list):
    with open(file, 'w') as f:
        for vertice in vertices:
            f.write(f"v {vertice[0]} {vertice[1]} {vertice[2]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        for mtl, faces, uvfaces in zip(mtls_list, faces_list, uvfaces_list):
            faces1 = faces + 1
            uvfaces1 = uvfaces + 1
            f.write(f"usemtl {mtl}\n")
            f.write(f"s off\n")
            for face, uvface in zip(faces1, uvfaces1):
                f.write(f"f {face[0]}/{uvface[0]} {face[1]}/{uvface[1]} {face[2]}/{uvface[2]}\n")

        f.write("\n")


def save_obj_with_vcolor(file, verts_colors, faces, uvs, uvfaces):
    with open(file, 'w') as f:
        for pos_color in verts_colors:
            f.write(f"v {pos_color[0]} {pos_color[1]} {pos_color[2]} {pos_color[3]} {pos_color[4]} {pos_color[5]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        faces1 = faces + 1
        uvfaces1 = uvfaces + 1
        for face, uvface in zip(faces1, uvfaces1):
            f.write(f"f {face[0]}/{uvface[0]} {face[1]}/{uvface[1]} {face[2]}/{uvface[2]}\n")

        f.write("\n")


def export_mpv_repr(out_root, cfg_file, cfg_file1, prefix="D:\\MSI_NB\\source\\data\\VideoLoops"):
    from config_parser import config_parser
    parser = config_parser()
    args = parser.parse_args(["--config", cfg_file, "--config1", cfg_file1])

    outpath = os.path.join(out_root, args.expname)
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
    ckpt_file = os.path.join(prefix, "logfinal", args.expname, "l5_epoch_0049.tar")
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

    def normalize_uv(uv, h, w):
        uv[:, 1] = -uv[:, 1]
        uv = uv * 0.5 + 0.5
        uv = uv * np.array([w - 1, h - 1]) / np.array([w, h]) + 0.5 / np.array([w, h])
        return uv

    def cull_unused(v, f):
        id_unique = np.unique(f)
        v_unique = v[id_unique]
        id_old2new = np.ones(len(v)).astype(id_unique.dtype) * -1
        id_old2new[id_unique] = np.arange(len(v_unique))
        newf = id_old2new[f]
        return v_unique, newf

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
    export_mpv_repr(
        "D:\\MSI_NB\\source\\data\\VideoLoops\\meshes",
        "configs/mpvgpnn_shared.txt",
        "configs/mpvgpnn_final/108fall1narrow_mpvgpnn.txt",
    )

