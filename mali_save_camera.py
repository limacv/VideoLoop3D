import numpy as np
import trimesh


def mali_save_camera_mesh(path: str, extrinsic, intrinsic=None, isc2w=True, camera_size="auto", near_far=None):
    """
    Args:
        path: path of the trimesh
        extrinsic:  R | T    or     -R^T | -R^T C, can be shape of [3/4, 4], [B, 3/4, 4]
        intrinsic:  can be shape of [B, 3, 3] or [3, 3]
        isc2w: when true, extrnsic is the matrix that transform camera coordinate to world coordinate (R | T)
        camera_size: when "auto", size=minimum distance of camera / 2, when float, size=camera_size
        near_far: when not None, should be size of [B, 2], which means (near, far)
    Returns:
        None, will save to path
    """
    default_intrinsic = np.array([1000, 0, 500,
                                  0, 1000, 500,
                                  0, 0, 1.]).astype(np.float32)
    extrinsic = np.array(extrinsic).astype(np.float32)
    intrinsic = np.array(intrinsic).astype(np.float32) if intrinsic is not None else default_intrinsic

    # standardlizing the matrices
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, ...]
    assert extrinsic.ndim == 3, f"extrinsic should be of shape [B, 3/4, 4], but got {extrinsic.shape}"
    ncamera = len(extrinsic)
    if extrinsic.shape[1] == 3:
        extrinsic = np.concatenate([extrinsic, np.zeros_like(extrinsic[:, :1, :])], axis=1)
        extrinsic[:, -1, -1] = 1.
    assert extrinsic.shape[1:] == (4, 4), f"extrinsic should be of shape [B, 3/4, 4], but got {extrinsic.shape}"

    if intrinsic.ndim == 2:
        intrinsic = intrinsic[None, ...]
    assert intrinsic.ndim == 3, f"intrinsic should be of shape [B, 3, 3], but got {intrinsic.shape}"

    intrinsic = np.broadcast_to(intrinsic, (ncamera, 3, 3))

    # inverse the extrinsic
    if not isc2w:
        poses = np.linalg.inv(extrinsic)
    else:
        poses = extrinsic

    # figure out the camera scale
    if camera_size == 'auto':
        camera_pos = poses[:, :3, 3]
        distance = camera_pos[:, None, :] - camera_pos[None, :, :]
        distance = np.linalg.norm(distance, axis=-1)
        distance = distance[distance > 0]
        distance = np.sort(distance)
        camera_size = distance[len(distance) // 10] * 0.5
        print(f"camera_size = {camera_size}")

    camera_size = float(camera_size)
    assert isinstance(camera_size, float), "camera_size should be auto or float"

    # the canonical camera
    camera_faces = [
        [0, 1, 2],
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 1],
        [5, 6, 7],
        [8, 9, 10],
    ]
    if near_far is not None:
        assert len(near_far) == len(extrinsic), "near_far should have save len as extrinsic"
        camera_faces += [
            [11, 12, 13],
            [14, 15, 16]
        ]
    camera_faces = np.array(camera_faces).astype(np.int32)
    camera_color = np.ones_like(camera_faces).astype(np.float32)

    all_vertices = []
    vertices_count = 0
    all_faces = []
    all_color = []

    for idx, (intrin_, pose_) in enumerate(zip(intrinsic, poses)):
        cx_fx = intrin_[0, 2] / intrin_[0, 0]
        cy_fy = intrin_[1, 2] / intrin_[1, 1]
        camera_vertices = [
            [0., 0., 0.],
            [-cx_fx, -cy_fy, 1],  # tl
            [cx_fx, -cy_fy, 1],  # tr
            [-cx_fx, cy_fy, 1],  # bl
            [cx_fx, cy_fy, 1],  # br

            # tops
            [-cx_fx * 0.8, -cy_fy * 1.1, 1],
            [cx_fx * 0.8, -cy_fy * 1.1, 1],
            [0., -cy_fy * 1.5, 1],

            # right
            [cx_fx, -cy_fy * 0.1, 1],
            [cx_fx, cy_fy * 0.1, 1],
            [cx_fx * 1.4, 0, 1]
        ]
        if near_far is not None:
            near, far = near_far[idx] / camera_size
            camera_vertices += [
                [-0.5, -0.5, near],
                [0.5, -0.5, near],
                [0, 0.5, near],
                [-0.5, -0.5, far],
                [0.5, -0.5, far],
                [0, 0.5, far],
            ]

        camera_vertices = np.array(camera_vertices).astype(np.float32) * camera_size
        camera_vertices = np.concatenate([camera_vertices, np.ones_like(camera_vertices[:, :1])], axis=-1)
        camera_vertices = pose_[None, ...] @ camera_vertices[..., None]
        camera_vertices = camera_vertices[:, :3, 0] / camera_vertices[:, 3:, 0]

        color = camera_color * idx / (ncamera - 1) * 255
        color = color.astype(np.uint8)

        all_vertices.append(camera_vertices.copy())
        all_color.append(color.copy())
        all_faces.append(camera_faces + vertices_count)
        vertices_count += len(camera_vertices)

    all_vertices = np.concatenate(all_vertices, axis=0)
    all_color = np.concatenate(all_color, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)
    mesh = trimesh.Trimesh(vertices=all_vertices,
                           faces=all_faces,
                           face_colors=all_color)
    ret = mesh.export(path, file_type=path.split('.')[-1])
    with open(path, "wb+") as f:
        f.write(ret)


if __name__ == "__main__":
    datadir = "./fern"

    import sys

    sys.path.append("D:\\MSI_NB\\source\\repos\\VideoLoop3D")
    import os
    from dataloader import load_llff_data

    images, poses, intrins, bds, render_poses, render_intrins = load_llff_data(basedir="./fern",
                                                                               factor=8,
                                                                               recenter=True)

    mali_save_camera_mesh(os.path.join(datadir, "visualize_camera.ply"),
                          poses, intrins
                          )
    mali_save_camera_mesh(os.path.join(datadir, "visualize_render_camera.ply"),
                          render_poses, render_intrins, camera_size=0.01
                          )
