import os
import cv2
import json
import glob
import imageio
import numpy as np


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    # img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
    #         if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        factor = factor
    else:
        factor = 1

    poses[:2, 4, :] = poses[:2, 4, :] / factor  # hw
    poses[2, 4, :] = poses[2, 4, :] / factor  # intrin

    if not load_imgs:
        return poses, bds, None

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=(1, 1), spherify=False, path_epi=False,
                   load_img=True, render_frm=120, render_scaling=1.):
    poses, bds, imgs = _load_data(basedir, factor=factor, load_imgs=load_img)
    # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    # for debug only
    # selected_idx = [0, 1, 2, 9, 8, 7, 10, 11, 19, 18]
    # imgs = imgs[..., selected_idx]
    # poses = poses[..., selected_idx]
    # bds = bds[..., selected_idx]

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32) if imgs is not None else None
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    bds = np.array([bds.min(), bds.max()]).astype(poses.dtype)
    sc = 1. / bds[0]
    poses[:, :3, 3] *= sc
    bds *= sc
    if bd_factor is not None:
        bds *= bd_factor

    if recenter:
        poses = recenter_poses(poses)

    # generate render_poses for video generation
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.abs(tt).max(0) * 0.8 * render_scaling
        # rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = render_frm
        N_rots = 2
        # Generate poses for spiral path
        # rads = [0.7, 0.2, 0.7]
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zrate=.5, zdelta=zdelta, rots=N_rots, N=N_views)

        if path_epi:
            #             zloc = np.percentile(tt, 10, 0)[2]
            rads[0] = rads[0] / 2
            render_poses = render_path_epi(c2w_path, up, rads[0], N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    poses = poses.astype(np.float32)

    H, W, focal = poses[:, :3, -1].T
    poses = poses[:, :3, :4]
    intrins = np.zeros_like(poses[:, :3, :3])
    intrins[:, -1, -1] = 1
    intrins[:, 0, 0] = focal
    intrins[:, 1, 1] = focal
    intrins[:, 0, 2] = 0.5 * W
    intrins[:, 1, 2] = 0.5 * H

    render_intrins = np.repeat(intrins[:1, ...], len(render_poses), 0)
    return imgs, poses, intrins, bds, render_poses, render_intrins


def load_mv_videos(basedir, factor=1, recenter=True, bd_factor=(1, 1), render_frm=120, render_scaling=1):
    _, poses, intrins, bds, render_poses, render_intrins = load_llff_data(basedir, factor, recenter,
                                                                          bd_factor=bd_factor,
                                                                          load_img=False,
                                                                          render_frm=render_frm,
                                                                          render_scaling=render_scaling)
    videos_path = sorted(glob.glob(basedir + f"/videos_{factor}/*"))
    videos = [imageio.mimread(vp, memtest=False) for vp in videos_path]
    cap = cv2.VideoCapture(videos_path[0])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    return videos, fps, poses, intrins, bds, render_poses, render_intrins


def load_masks(imgpaths):
    msklist = []

    for imgdir in imgpaths:
        mskdir = imgdir.replace('images', 'masks').replace('.jpg', '.png')
        msk = imageio.imread(mskdir)

        H, W = msk.shape[0:2]
        msk = msk / 255.0

        # msk   = np.sum(msk, axis=2)
        # msk[msk < 3.0] = 0.0
        # msk[msk == 3.0] = 1.0
        # msk = 1.0 - msk

        newmsk = np.zeros((H, W), dtype=np.float32)
        newmsk[np.logical_and((msk[:, :, 0] == 0), (msk[:, :, 1] == 0), (msk[:, :, 2] == 1.0))] = 1.0

        # imageio.imwrite('newmsk.png', newmsk)
        # print(imgpaths, mskdir, H, W)
        # print(sss)

        msklist.append(newmsk)

    msklist = np.stack(msklist, 0)

    return msklist


def has_matted(imgpaths):
    exampledir = imgpaths[-1].replace('images', 'images_rgba').replace('.jpg', '.png')
    return os.path.exists(exampledir)


def load_matted(imgpaths):
    imglist = []
    for imgdir in imgpaths:
        imgdir = imgdir.replace('images', 'images_rgba').replace('.jpg', '.png')
        rgba = imageio.imread(imgdir)
        assert rgba.shape[-1] == 4, "cannot load rgba png"
        rgba = rgba / 255.0
        rgba[..., :3] = rgba[..., :3] * rgba[..., 3:4]
        imglist.append(rgba)

    imglist = np.stack(imglist, 0)
    return imglist


def load_images(imgpaths):
    imglist = []

    for imgdir in imgpaths:
        img = imageio.imread(imgdir)
        img = img / 255.0
        imglist.append(img)

    imglist = np.stack(imglist, 0)

    return imglist


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def render_path_spiral(c2w, up, rads, focal, zrate, zdelta, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # view direction
        # c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), (np.cos(theta * zrate) * zdelta) ** 2, 1.]) * rads)
        # camera poses
        z = normalize(np.array([0, 0, focal] - c))
        render_poses.append(viewmatrix(z, up, c))
    return np.stack(render_poses)
