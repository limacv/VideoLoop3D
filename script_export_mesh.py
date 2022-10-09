import torch
import os
import numpy as np
import imageio


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


path = "D:\\MSI_NB\\source\\data\\VideoLoops\\logs\\sparse_ustfallclose_f50_v02Gpnnlm0PtSt5321_otherGpnnlm100PtSt3321_roun2s01\\l5_epoch_0049.tar"


outpath = os.path.join("D:\\MSI_NB\\source\\data\\VideoLoops\\meshes",
                       os.path.basename(os.path.dirname(path)))
os.makedirs(outpath, exist_ok=True)
save_split = True  # if true, save to separte files

state_dict = torch.load(path)
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


def normalize_uv(uv, h, w):
    uv = uv * 0.5 + 0.5
    uv = uv * np.array([w - 1, h - 1]) / np.array([w, h]) + 0.5 / np.array([w, h])
    return uv


# TODO: culling unused vertices
def cull_unused(v, f):
    return v, f


uvs_static = normalize_uv(uvs_static, atlas_h_static, atlas_w_static)
uvs_dynamic = normalize_uv(uvs_dynamic, atlas_h_dynamic, atlas_w_dynamic)


if save_split:
    with open(os.path.join(outpath, "meta.txt"), 'w') as f:
        f.write("fps = 25\n")
        f.write(f"frame_num = {frame_num}\n")

    # save static
    newv, newf = cull_unused(verts, faces_static)
    newuv, newuvf = cull_unused(uvs_static, uvfaces_static)
    save_obj_multimaterial(os.path.join(outpath, "static.obj"),
                           newv, [newf], newuv, [newuvf], ['static_texture'])
    imageio.imwrite(os.path.join(outpath, "static.png"), atlas_static)

    newv, newf = cull_unused(verts, faces_dynamic)
    newuv, newuvf = cull_unused(uvs_dynamic, uvfaces_dynamic)
    save_obj_multimaterial(os.path.join(outpath, "dynamic.obj"),
                           newv, [newf], newuv, [newuvf], ['dynamic_texture'])
    vidoutpath = os.path.join(outpath, "dynamic")
    os.makedirs(vidoutpath, exist_ok=True)
    for i in range(frame_num):
        imageio.imwrite(os.path.join(vidoutpath, f"{i:04d}.png"),
                        atlas_dynamic[i])
