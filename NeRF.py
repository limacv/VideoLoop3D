import torch
import torch.nn as nn
import torch.nn.functional as torchf
import os
import imageio
import time
import cv2
from utils import *
from NeRF_modules import *

activate = {'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'none': lambda x: x,
            'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
            'softplus': lambda x: nn.Softplus()(x - 1),
            'tanh': torch.tanh}


class NeRFModulateT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires)

        self.input_ch_views = 0
        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views)

        self.output_ch = 5 if args.N_importance > 0 else 4

        # Create initial latent code
        self.embedtime_fn, self.input_ch_times = get_embedder(args.time_multires, args.embed_type,
                                                              dict_len=args.time_len,
                                                              latent_size=args.latent_size)

        skips = [4]
        self.mlp_coarse = NeRFmlp(
            D=args.netdepth, W=args.netwidth,
            input_ch=self.input_ch, output_ch=self.output_ch, skips=skips, input_ch_latent_t=self.input_ch_times,
            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

        if args.use_two_models_for_fine > 0:
            self.mlp_fine = NeRFmlp(
                D=args.netdepth, W=args.netwidth,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips, input_ch_latent_t=self.input_ch_times,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)
        else:
            self.mlp_fine = self.mlp_coarse

        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = activate['none']
        self.r2o = self.raw2outputs_old if args.use_raw2outputs_old else self.raw2outputs

    def mlpforward(self, inputs, viewdirs, times, mlp, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        if times is not None:
            input_time_flat = times[:, None].expand_as(inputs[..., :1]).reshape(-1, 1)
            embedded_time = self.embedtime_fn(input_time_flat)
            embedded = torch.cat([embedded, embedded_time], -1)

        # batchify execution
        if netchunk is None:
            outputs_flat = mlp(embedded)
        else:
            outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = (z_vals[..., 1:] - z_vals[..., :-1]).abs()  # [N_rays, N_samples - 1]
        # dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., 3]) * raw_noise_std
            noise = noise[..., :-1]
            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        # if not self.training and self.args.render_rmnearplane > 0:
        #     mask = z_vals[:, 1:]
        #     mask = mask > self.args.render_rmnearplane / 128
        #     mask = mask.type_as(density)
        #     density = mask * density

        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # alpha = raw2alpha(raw[..., :-1, 3] + noise, dists, act_fn=self.sigma_activate)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None]) * (28 / 255)

        return rgb_map, density, acc_map, weights, depth_map

    def raw2outputs_old(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = (z_vals[..., 1:] - z_vals[..., :-1]).abs()
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None]) * (28 / 255)

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_rays(self,
                    ray_batch,
                    ray_infos,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 6 else None
        near, far, times = torch.split(ray_infos.reshape(-1, 3), 1, dim=-1)
        miss_mask = far <= near
        near[miss_mask] = 2
        far[miss_mask] = 4  # really don't know why leads to the nan
        intersec_mask = torch.logical_not(miss_mask).float()

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o) * perturb

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw = self.mlpforward(pts, viewdirs, times, self.mlp_coarse)

        rgb_map, density_map, acc_map, weights, depth_map = self.r2o(raw, z_vals, rays_d, raw_noise_std,
                                                                     white_bkgd, pytest=pytest)
        rgb_map = rgb_map * intersec_mask
        acc_map = acc_map * intersec_mask[..., 0]
        weights = weights * intersec_mask
        depth_map = depth_map * intersec_mask[..., 0]

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0 = rgb_map, depth_map, acc_map, density_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()
            if self.args.N_samples_fine < z_vals.shape[-1]:
                choice = np.random.choice(z_vals.shape[-1], self.args.N_samples_fine, replace=False)
                z_vals = z_vals[:, choice]
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            raw = self.mlpforward(pts, viewdirs, times, mlp)

            rgb_map, density_map, acc_map, weights, depth_map = self.r2o(raw, z_vals, rays_d, raw_noise_std,
                                                                         white_bkgd, pytest=pytest)
            rgb_map = rgb_map * intersec_mask
            acc_map = acc_map * intersec_mask[..., 0]
            weights = weights * intersec_mask
            depth_map = depth_map * intersec_mask[..., 0]

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'density_map': density_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if "density" in k:
                continue
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, t=None, chunk=1024 * 32, intrinsics=None, rays=None, poses=None,
                **kwargs):
        """
        render rays or render poses, rays and poses sould atleast specify one
        """
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"
            rays = rays.reshape(-1, 2, 3).permute(1, 0, 2)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **kwargs)
            rgb0, acc0, depth0 = extras['rgb0'], extras['acc0'], extras["depth0"]
            others = {}
            rgb = torch.cat([self.tonemapping(rgb), acc[..., None]], dim=-1)
            rgb0 = torch.cat([self.tonemapping(rgb0), acc0[..., None]], dim=-1)
            return rgb, rgb0, others

        #  evaluation
        else:
            assert poses is not None and intrinsics is not None, "Please specify poses when in the eval model"
            if t is None:
                rgbs, depths = self.render_path_vary_t(H, W, chunk, poses, intrinsics, **kwargs)
            else:
                rgbs, depths = self.render_path_fix_t(H, W, t, chunk, poses, intrinsics, **kwargs)
            return self.tonemapping(rgbs), depths

    def render(self, H, W, rays=None, t=None, box=None,
               ndc=False, chunk=1024,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays

        if use_viewdirs:
            # provide ray directions as inputs
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays_tensor(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        # if ndc:
        #     # for forward facing scenes
        #     rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        t = t * torch.ones_like(rays_d[..., :1])

        # find near far by intersection
        with torch.no_grad():
            box = torch.tensor(box).type_as(rays_o)
            rays_d_denorm = torch.reciprocal(rays_d).clamp(-1e10, 1e10)
            intersec1 = (box[0:1] - rays_o) * rays_d_denorm
            intersec2 = (box[1:2] - rays_o) * rays_d_denorm
            near = torch.min(intersec1, intersec2).max(dim=-1, keepdim=True)[0].clamp_min(0)
            far = torch.max(intersec1, intersec2).min(dim=-1, keepdim=True)[0]
            far = torch.max(near, far)

        rays = torch.cat([rays_o, rays_d], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)
        rays_info = torch.cat([near, far, t], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], rays_info[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path_fix_t(self, H, W, t, chunk, render_poses, intrinsics, render_kwargs):
        rgbs = []
        depths = []

        print("rendering frame: ", end='')
        for i, (c2w, K) in enumerate(zip(render_poses, intrinsics)):
            print(f"{i}", end=' ', flush=True)
            rays = get_rays_tensor(H, W, K, c2w)
            rays = torch.stack(rays, dim=0)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_path_vary_t(self, H, W, chunk, render_poses, intrinsics, render_kwargs):
        rgbs = []
        depths = []

        print("rendering frame: ", end='')
        for i in range(render_poses.shape[0]):
            print(f"{i}", end=' ', flush=True)
            c2w = render_poses[i]
            K = intrinsics[i]
            t = i % self.args.time_len
            rays = get_rays_tensor(H, W, K, c2w)
            rays = torch.stack(rays, dim=0)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths


class NeRFTemporal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires)
        self.embedtime_fn, self.input_ch_time = get_embedder(args.time_multires, args.time_embed_type,
                                                             dict_len=args.time_len,
                                                             latent_size=args.latent_size)
        if args.ambient_slicing_dim > 0:
            self.embedslice_fn, self.input_ch_slice = get_embedder(args.slice_multires,
                                                                   input_dim=args.ambient_slicing_dim)
        else:
            self.embedslice_fn, self.input_ch_slice = None, 0

        self.input_ch_views = 0
        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views)

        self.output_ch = 5 if args.N_importance > 0 else 4

        skips = [4]
        self.mlp_coarse = NeRFmlp(
            D=args.netdepth, W=args.netwidth,
            input_ch=self.input_ch, input_ch_latent_t=self.input_ch_slice, output_ch=self.output_ch, skips=skips,
            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)
        self.dmlp_coarse = GeneralMLP(
            D=args.dnetdepth, W=args.dnetwidth,
            input_ch=self.input_ch, input_ch_time=self.input_ch_time, skips=skips
        )
        if args.ambient_slicing_dim > 0:
            self.slicemlp_coarse = GeneralMLP(
                D=args.slicenetdepth, W=args.slicenetwidth,
                input_ch=self.input_ch, input_ch_time=self.input_ch_time, skips=skips,
                output_ch=args.ambient_slicing_dim
            )
        else:
            self.slicemlp_coarse = None
        self.slicemlp_fine = self.slicemlp_coarse

        if args.use_two_models_for_fine:
            self.mlp_fine = NeRFmlp(
                D=args.netdepth, W=args.netwidth,
                input_ch=self.input_ch, input_ch_latent_t=self.input_ch_slice, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)
        else:
            self.mlp_fine = self.mlp_coarse

        self.dmlp_fine = None
        if args.use_two_dmodels_for_fine:
            self.dmlp_fine = GeneralMLP(
                D=args.dnetdepth, W=args.dnetwidth,
                input_ch=self.input_ch, input_ch_time=self.input_ch_time, skips=skips
            )
        else:
            self.dmlp_fine = self.dmlp_coarse

        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = activate['none']
        self.r2o = self.raw2outputs_old if args.use_raw2outputs_old else self.raw2outputs
        self.render_canonical = args.render_canonical

    def mlpforward(self, inputs, viewdirs, times, dmlp, mlp, smlp=None, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        d_embedded = self.embed_fn(inputs_flat)

        input_time_flat = times[:, None].expand_as(inputs[..., :1]).reshape(-1, 1)
        d_embedded_time = self.embedtime_fn(input_time_flat)
        d_embedded = torch.cat([d_embedded, d_embedded_time], -1)

        netchunk1 = int(netchunk * 1.4)
        # batchify execution
        if self.render_canonical:
            dx = torch.zeros_like(inputs_flat)
        else:
            dx = torch.cat([dmlp(d_embedded[i:i + netchunk1])
                            for i in range(0, d_embedded.shape[0], netchunk1)], 0)

        if smlp is not None:
            slic = torch.cat([smlp(d_embedded[i:i + netchunk1])
                              for i in range(0, d_embedded.shape[0], netchunk1)], 0)
            assert self.embedslice_fn is not None
            embedded_slice = self.embedslice_fn(slic)
        else:
            embedded_slice = None

        embedded = self.embed_fn(inputs_flat + dx)
        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)
        if smlp is not None:
            embedded = torch.cat([embedded, embedded_slice], -1)

        # batchify execution
        outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        dx = dx.reshape(list(inputs.shape[:-1]) + [dx.shape[-1]])
        return outputs, dx

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = (z_vals[..., 1:] - z_vals[..., :-1]).abs()  # [N_rays, N_samples - 1]
        # dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., 3]) * raw_noise_std
            noise = noise[..., :-1]
            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        # if not self.training and self.args.render_rmnearplane > 0:
        #     mask = z_vals[:, 1:]
        #     mask = mask > self.args.render_rmnearplane / 128
        #     mask = mask.type_as(density)
        #     density = mask * density

        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # alpha = raw2alpha(raw[..., :-1, 3] + noise, dists, act_fn=self.sigma_activate)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None]) * (28 / 255)

        return rgb_map, density, acc_map, weights, depth_map

    def raw2outputs_old(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = (z_vals[..., 1:] - z_vals[..., :-1]).abs()
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None]) * (28 / 255)

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_rays(self,
                    ray_batch,
                    ray_infos,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 6 else None
        near, far, times = torch.split(ray_infos.reshape(-1, 3), 1, dim=-1)
        miss_mask = far <= near
        near[miss_mask] = 2
        far[miss_mask] = 4  # really don't know why leads to the nan
        intersec_mask = torch.logical_not(miss_mask).float()

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw, dx = self.mlpforward(pts, viewdirs, times, self.dmlp_coarse, self.mlp_coarse, self.slicemlp_coarse)

        rgb_map, density_map, acc_map, weights, depth_map = self.r2o(raw, z_vals, rays_d, raw_noise_std,
                                                                     white_bkgd, pytest=pytest)
        rgb_map = rgb_map * intersec_mask
        acc_map = acc_map * intersec_mask[..., 0]
        weights = weights * intersec_mask
        depth_map = depth_map * intersec_mask[..., 0]

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0 = rgb_map, depth_map, acc_map, density_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()
            if self.args.N_samples_fine < z_vals.shape[-1]:
                choice = np.random.choice(z_vals.shape[-1], self.args.N_samples_fine, replace=False)
                z_vals = z_vals[:, choice]
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            raw, dx = self.mlpforward(pts, viewdirs, times, self.dmlp_fine, self.mlp_fine, self.slicemlp_coarse)

            rgb_map, density_map, acc_map, weights, depth_map = self.r2o(raw, z_vals, rays_d, raw_noise_std,
                                                                         white_bkgd, pytest=pytest)
            rgb_map = rgb_map * intersec_mask
            acc_map = acc_map * intersec_mask[..., 0]
            weights = weights * intersec_mask
            depth_map = depth_map * intersec_mask[..., 0]

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'density_map': density_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if "density" in k:
                continue
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, t=None, chunk=1024 * 32, intrinsics=None, rays=None, poses=None,
                **kwargs):
        """
        render rays or render poses, rays and poses sould atleast specify one
        """
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"

            rays = rays.reshape(-1, 2, 3).permute(1, 0, 2)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **kwargs)
            rgb0, acc0, depth0 = extras['rgb0'], extras['acc0'], extras["depth0"]
            others = {}
            # rgb to rgba
            rgb = torch.cat([self.tonemapping(rgb), acc[..., None]], dim=-1)
            rgb0 = torch.cat([self.tonemapping(rgb0), acc0[..., None]], dim=-1)
            return rgb, rgb0, others

        #  evaluation
        else:
            assert poses is not None and intrinsics is not None, "Please specify poses when in the eval model"
            if t is None:
                rgbs, depths = self.render_path_vary_t(H, W, chunk, poses, intrinsics, **kwargs)
            else:
                rgbs, depths = self.render_path_fix_t(H, W, t, chunk, poses, intrinsics, **kwargs)
            return self.tonemapping(rgbs), depths

    def render(self, H, W, rays=None, t=None, box=None,
               ndc=False, chunk=1024,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays

        if use_viewdirs:
            # provide ray directions as inputs
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays_tensor(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        # if ndc:
        #     # for forward facing scenes
        #     rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        t = 0 if t is None else t
        t = t * torch.ones_like(rays_d[..., :1])
        # near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        # find near far by intersection
        with torch.no_grad():
            box = torch.tensor(box).type_as(rays_o)
            rays_d_denorm = torch.reciprocal(rays_d).clamp(-1e10, 1e10)
            intersec1 = (box[0:1] - rays_o) * rays_d_denorm
            intersec2 = (box[1:2] - rays_o) * rays_d_denorm
            near = torch.min(intersec1, intersec2).max(dim=-1, keepdim=True)[0].clamp_min(0)
            far = torch.max(intersec1, intersec2).min(dim=-1, keepdim=True)[0]
            far = torch.max(near, far)
        # print(f"near min {near.min()}, near max {near.max()}, far min {far.min()}, far max {far.max()}")
        rays = torch.cat([rays_o, rays_d], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)
        rays_info = torch.cat([near, far, t], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], rays_info[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path_fix_t(self, H, W, t, chunk, render_poses, intrinsics, render_kwargs):
        rgbs = []
        depths = []

        print("rendering frame: ", end='')
        for i, (c2w, K) in enumerate(zip(render_poses, intrinsics)):
            print(f"{i}", end=' ', flush=True)
            rays = get_rays_tensor(H, W, K, c2w)
            rays = torch.stack(rays, dim=0)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_path_vary_t(self, H, W, chunk, render_poses, intrinsics, render_kwargs):
        rgbs = []
        depths = []

        print("rendering frame: ", end='')
        for i in range(render_poses.shape[0]):
            print(f"{i}", end=' ', flush=True)
            c2w = render_poses[i]
            K = intrinsics[i]
            t = i % self.args.time_len
            rays = get_rays_tensor(H, W, K, c2w)
            rays = torch.stack(rays, dim=0)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths


class NeUVFModulateT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embeds = []
        # ###########################
        # handle all the embedding
        # Embedding for xyz input to the XYZ2UV
        self.embed_fn, self.input_ch = get_embedder(args.multires, embed_type=args.embed_type,
                                                    window_start=args.multires_window_start,
                                                    window_end=args.multires_window_end,
                                                    log2_hash_size=args.log2_embed_hash_size)
        self.embeds.append(self.embed_fn)

        # Embedding for time input into XYZ2UV
        self.embedtime_fn, self.input_ch_time = get_embedder(args.time_multires, embed_type=args.time_embed_type,
                                                             dict_len=args.time_len,
                                                             latent_size=args.latent_size)
        self.embeds.append(self.embedtime_fn)

        # Embedding for view input into Texture MLP
        self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views,
                                                              window_start=args.multires_views_window_start,
                                                              window_end=args.multires_views_window_end)
        self.embeds.append(self.embeddirs_fn)

        # Embedding for time input into Texture MLP
        if args.use_two_time_for_tex:
            self.embedtime_fn_tex, self.input_ch_time_tex = get_embedder(
                args.time_multires_for_tex, embed_type=args.time_embed_type_for_tex,
                dict_len=args.time_len,
                window_start=args.time_multires_window_start_for_tex,
                window_end=args.time_multires_window_end_for_tex,
                latent_size=args.latent_size_for_tex)
            self.embeds.append(self.embedtime_fn_tex)
        else:
            self.embedtime_fn_tex, self.input_ch_time_tex = self.embedtime_fn, self.input_ch_time

        # #########################
        # create networks
        self.explicit_warp_type = args.explicit_warp_type
        if self.explicit_warp_type == 'none':
            self.explicit_warp = None
        elif self.explicit_warp_type == 'rigid':
            self.explicit_warp = WarpRigid(args.time_len, args.uv_gts, args.t2uv_gt_id)
        elif self.explicit_warp_type == 'proj':
            self.explicit_warp = WarpProj(args.time_len, args.uv_gts, args.t2uv_gt_id, args.canonicaldir)
        elif self.explicit_warp_type == 'prnet':
            self.explicit_warp = WarpKpt(args.time_len, args.uv_gts, args.t2uv_gt_id, args.canonicaldir, args.kptidsdir)
        elif self.explicit_warp_type == 'kptaffine':
            self.explicit_warp = WarpKptAdvanced(args.uv_gts, args.t2uv_gt_id, args.canonicaldir, args.kptidsdir,
                                                 args.model_affine, args.rbf_perframe)
        else:
            raise RuntimeError(f"Not implement error {self.explicit_warp}")

        # MLPs for XYZ2UV
        skips = [args.netdepth // 2] if args.netdepth > 4 else []
        self.output_ch = 4  # uvw + sigma
        self.mlp_coarse = NeRFmlp(
            D=args.netdepth, W=args.netwidth,
            input_ch=self.input_ch, output_ch=self.output_ch, skips=skips, input_ch_latent_t=self.input_ch_time,
            input_ch_views=0, use_viewdirs=False)

        if args.use_two_models_for_fine:
            self.mlp_fine = NeRFmlp(
                D=args.netdepth, W=args.netwidth,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips, input_ch_latent_t=self.input_ch_time,
                input_ch_views=0, use_viewdirs=False)
        else:
            self.mlp_fine = self.mlp_coarse

        # MLPs for texture map
        self.texture_type = args.texture_type
        self.alpha_type = args.alpha_type
        self.embeduv_fn, self.input_ch_uv, self.texture_map_coarse = self.create_texture_map(args, "mlp")
        self.embeds.append(self.embeduv_fn)
        if args.use_two_texmodels_for_fine:
            _, _, self.texture_map = self.create_texture_map(args)
        else:
            self.texture_map = self.texture_map_coarse

        # MLPs for cycle loss
        self.uv2xyz_mlp, self.uv2xyz_mlp_fine = None, None
        if args.cycle_loss_weight > 0:
            if args.use_two_embed_for_cycle:
                self.embed_fn_cycle, self.input_ch_cycle = get_embedder(
                    args.multires_for_cycle, log2_hash_size=args.log2_hash_size_for_cycle)
                self.embeds.append(self.embed_fn_cycle)
            else:
                self.embed_fn_cycle, self.input_ch_cycle = self.embed_fn, self.input_ch
            if args.use_two_time_for_cycle:
                self.embedtime_fn_cycle, self.input_ch_time_cycle = get_embedder(
                    args.time_multires_for_cycle,
                    dict_len=args.time_len,
                    latent_size=args.latent_size_for_cycle,
                    embed_type=args.time_embed_type_for_cycle)
                self.embeds.append(self.embedtime_fn_cycle)
            else:
                self.embedtime_fn_cycle, self.input_ch_time_cycle = self.embedtime_fn, self.input_ch_time

            self.uv2xyz_mlp = GeneralMLP(
                D=args.cyclenetdepth, W=args.cyclenetwidth,
                input_ch=self.input_ch_cycle, output_ch=3, skips=skips,
                input_ch_time=self.input_ch_time_cycle, time_layer_idx=args.cyclenet_time_layeridx)
            if args.use_two_models_for_fine:
                self.uv2xyz_mlp_fine = GeneralMLP(
                    D=args.cyclenetdepth, W=args.cyclenetwidth,
                    input_ch=self.input_ch_cycle, output_ch=3, skips=skips,
                    input_ch_time=self.input_ch_time_cycle, time_layer_idx=args.cyclenet_time_layeridx)
            else:
                self.uv2xyz_mlp_fine = self.uv2xyz_mlp

        # about geometry and density
        self.density_type = args.density_type
        if args.sigma_activate == "volsdf":
            self.register_parameter("density_beta", nn.Parameter(torch.tensor(1.), requires_grad=True))
            self.register_parameter("density_alpha", nn.Parameter(torch.tensor(1.), requires_grad=True))

            def sdf2density(s, beta=self.density_beta, alpha=self.density_alpha):
                beta = beta.abs().to(s.device).clamp_min(0.1)
                alpha = alpha.abs().to(s.device).clamp_min(0.1)
                return torch.sigmoid(- s * beta) * alpha

            self.sigma_activate = sdf2density
        else:
            self.sigma_activate = activate[args.sigma_activate]

        self.rgb_activate = activate[args.rgb_activate]
        self.uv_activate = activate[args.uv_activate]
        self.tonemapping = activate['none']
        self.embeds = [embed_ for embed_ in self.embeds if hasattr(embed_, "update_activate_freq")]
        print(f"NeUVFmodulateT::find {len(self.embeds)} windowed embedders")
        # used only for get_texture()
        self._example_view_embed = torch.zeros(1, self.input_ch_views)
        self._example_ts_embed = torch.zeros(1, self.input_ch_time_tex)

        self.density_offset = None
        if len(args.uvweightdir) > 0:
            uv_weight_map = imageio.imread(args.uvweightdir)
            uv_weight_map = torch.tensor(uv_weight_map / 255)[None, None, :, :, 0].float()
            self.register_buffer("uv_weight_map", uv_weight_map)

    def create_texture_map(self, args, texture_type=None):
        texture_type = args.texture_type if texture_type is None else texture_type
        skips = [args.texnetdepth // 2, ]
        if texture_type == "map":
            embeduv_fn, input_ch_uv = lambda x: x, 2
            texture_map = TextureMap(
                resolution=args.texture_map_resolution,
                face_roi=args.uv_map_face_roi,
                output_ch=args.texture_map_channel,
                activate=activate[args.rgb_activate],  # for undo the activate while loading
                grad_multiply=args.texture_map_gradient_multiply
            )
            if len(args.texture_map_ini) > 0:
                print(f"Load texture map from {args.texture_map_ini}")
                texture_map.load(args.texture_map_ini)
        elif texture_type == "mlp":
            embeduv_fn, input_ch_uv = get_embedder(args.tex_multires,
                                                   args.tex_embed_type,
                                                   input_dim=2,
                                                   log2_hash_size=args.tex_log2_hash_size)
            texture_map = GeneralMLP(
                D=args.texnetdepth, W=args.texnetwidth, input_ch=input_ch_uv,
                input_ch_view=self.input_ch_views, input_ch_time=self.input_ch_time_tex,
                view_layer_idx=args.texnet_view_layeridx, time_layer_idx=args.texnet_time_layeridx,
                output_ch=3, skips=skips
            )
        elif texture_type == "fuse":
            embeduv_fn, input_ch_uv = get_embedder(args.tex_multires,
                                                   args.tex_embed_type,
                                                   input_dim=2,
                                                   log2_hash_size=args.tex_log2_hash_size)
            texture_map = TextureFuse(
                D=args.texnetdepth, W=args.texnetwidth, input_ch=input_ch_uv,
                input_ch_view=self.input_ch_views, input_ch_time=self.input_ch_time_tex,
                view_layer_idx=args.texnet_view_layeridx, time_layer_idx=args.texnet_time_layeridx,
                resolution=args.texture_map_resolution, face_roi=args.uv_map_face_roi,
                output_ch=3, activate=activate[args.rgb_activate],  # for undo the activate while loading
                uv_embedder=embeduv_fn,  # for converting mlp to texture map
                skips=skips,
                texture_map_gradient_multiply=args.texture_map_gradient_multiply
            )
        else:
            raise RuntimeError(f"texture_type of {texture_type} unrecognized")
        return embeduv_fn, input_ch_uv, texture_map

    def set_explicit_warp_grad(self, state: bool):
        if self.explicit_warp is None:
            return
        for params in self.explicit_warp.parameters():
            params.requires_grad = state

    def update_step(self, step):
        for embed_ in self.embeds:
            embed_.update_activate_freq(step)

        # convert to
        if step == self.args.promote_fuse_texture_step \
                and isinstance(self.texture_map, TextureFuse):
            self.texture_map.promote_texture()
            if id(self.texture_map) != id(self.texture_map_coarse) \
                    and isinstance(self.texture_map_coarse, TextureFuse):
                self.texture_map_coarse.promote_texture()

        if step == self.args.freeze_uvfield_step:
            print("Freezing the UV field")
            for parameters in self.mlp_fine.parameters():
                parameters.requires_grad = False
                parameters.grad = None
            for parameters in self.mlp_coarse.parameters():
                parameters.requires_grad = False
                parameters.grad = None
            for parameters in self.embedtime_fn.parameters():
                parameters.requires_grad = False
                parameters.grad = None

    def mlpforward(self, inputs_raw, viewdirs, times,
                   mlp, uv2rgb, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        if times.dim() != inputs_raw.dim():
            times = times[:, None].expand_as(inputs_raw[..., :1])
        else:
            times = times.expand_as(inputs_raw[..., :1])

        inputs_warp = self.explicit_warp(inputs_raw, times) if self.explicit_warp is not None else inputs_raw
        inputs = inputs_warp
        # get all embeddings --1st round
        # t1 = time.time()
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        input_time_flat = times.reshape(-1, 1)
        embedded_time = self.embedtime_fn(input_time_flat)
        embedded = torch.cat([embedded, embedded_time], -1)
        # t2 = time.time()
        # batchify execution
        if netchunk is None:
            outputs_flat = mlp(embedded)
        else:
            outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)
        # t3 = time.time()

        # print(f"embed time = {t2 - t1}, mlp time = {t3 - t2}, total = {t3 - t1}")
        xyz_flat = outputs_flat[:, :3] + inputs_flat
        xyz_flat_norm = xyz_flat.norm(dim=-1, keepdim=True)
        xyz_flat = xyz_flat / xyz_flat_norm
        uv_flat = xyz2uv_stereographic(xyz_flat, normalized=True)
        uv_flat = self.uv_activate(uv_flat)
        uv = torch.reshape(uv_flat, list(inputs.shape[:-1]) + [uv_flat.shape[-1]])

        if self.density_type == 'direct':
            density_raw = outputs_flat[:, 3:]
        elif self.density_type == 'xyz_norm':
            density_raw = xyz_flat_norm.clamp_min(1e-10) - 0.5  # 0.5 is the canonical ball radius
        else:
            raise RuntimeError(f"density type {self.density_type} not recognized")

        # this only happens during testing
        if self.density_offset is not None:
            density_offset = self.density_offset(uv_flat)
            density = density_raw + density_offset
        else:
            density = density_raw

        # for texture mlp or texture map
        if uv2rgb is not None:
            # figure out embeddings
            embedded_uv = self.embeduv_fn(uv_flat)

            if viewdirs.dim() != inputs.dim():
                input_dirs = viewdirs[:, None].expand(inputs.shape)
            else:
                input_dirs = viewdirs.expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)

            if id(self.embedtime_fn_tex) != id(self.embedtime_fn):
                embedded_time4tex = self.embedtime_fn_tex(input_time_flat)
            else:
                embedded_time4tex = embedded_time

            tex_embedded = torch.cat([embedded_uv, embedded_dirs, embedded_time4tex], dim=-1)
            self._example_view_embed = embedded_dirs[:1, :]
            self._example_ts_embed = embedded_time4tex[:1, :]
            texture_out_flat = torch.cat([uv2rgb(tex_embedded[i:i + netchunk])
                                          for i in range(0, tex_embedded.shape[0], netchunk)], 0)
            if texture_out_flat.shape[-1] > 3:
                alphas = texture_out_flat[..., 3:]
                alphas = alphas.reshape(list(inputs.shape[:-1]) + [alphas.shape[-1]])
            else:
                alphas = None
        else:
            texture_out_flat = torch.ones_like(density[..., :0])
            alphas = None

        outputs_flat = torch.cat([texture_out_flat, density], dim=-1)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        outputs_xyz = torch.reshape(xyz_flat, list(inputs.shape[:-1]) + [xyz_flat.shape[-1]])
        return outputs, uv, inputs_warp, outputs_xyz, alphas

    def forward(self, H, W, t=None, chunk=1024 * 32, intrinsics=None, rays=None, poses=None, pts_viewdir=None,
                **kwargs):
        """
        render rays or render poses, rays and poses sould atleast specify one
        """
        if pts_viewdir is not None:
            pts, viewdir = pts_viewdir.split([3, 3], dim=-1)
            latent_t = t * torch.ones_like(pts[..., :1])
            density_raw0, uv0, _, _, _ = self.mlpforward(pts, viewdir, latent_t,
                                                         self.mlp_coarse, None, chunk)
            density_raw, uv, _, _, _ = self.mlpforward(pts, viewdir, latent_t,
                                                       self.mlp_fine, None, chunk)
            return uv, uv0, density_raw, density_raw0
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"
            rays = rays.reshape(-1, 2, 3).permute(1, 0, 2)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **kwargs)
            rgb0, acc0, depth0 = extras['rgb0'], extras['acc0'], extras["depth0"]
            others = {}

            ###############
            # compute other losses here to accelerate multi-GPU
            pts, pts0 = extras['pts'], extras['pts0']
            weight, weight0 = extras['weight_map'], extras['weight0']
            alpha = extras.get('alpha', None)

            if self.args.sparsity_loss_weight > 0:
                others["sparsity"] = self.compute_sparsity_loss(weight).reshape(1, 1)

            if self.args.cycle_loss_weight > 0:
                pts_warp, pts_warp0 = extras.get('pts_warp', None), extras.get('pts_warp0', None)
                pts_out, pts_out0 = extras.get('pts_out', None), extras.get('pts_out0', None)
                weight0_s, topk_id0 = torch.topk(weight0[:, :-1], 6, dim=-1)
                weight_s, topk_id = torch.topk(weight[:, :-1], 6, dim=-1)
                topk_id0 = topk_id0.reshape(-1, 6, 1).expand(-1, -1, 3)
                topk_id = topk_id.reshape(-1, 6, 1).expand(-1, -1, 3)
                pts_warp0_s = torch.gather(pts_warp0, 1, topk_id0)
                pts_out0_s = torch.gather(pts_out0, 1, topk_id0)
                pts_warp_s = torch.gather(pts_warp, 1, topk_id)
                pts_out_s = torch.gather(pts_out, 1, topk_id)
                cycle_loss0 = self.compute_cycle_loss(pts_warp0_s, pts_out0_s, t, self.uv2xyz_mlp, weight0_s)
                cycle_loss = self.compute_cycle_loss(pts_warp_s, pts_out_s, t, self.uv2xyz_mlp_fine, weight_s)
                others["cycle"] = (cycle_loss + cycle_loss0).reshape(1, 1)

            if self.args.alpha_loss_weight > 0:
                alpha_loss = self.compute_alpha_loss(alpha, weight)
                # alpha_loss0 = self.compute_alpha_loss(extras.get("alpha0", None))
                others["alpha"] = alpha_loss.reshape(1, 1).type_as(rgb)

            if self.args.smooth_loss_weight > 0:
                rgb_left, rgb_right = rgb.reshape(-1, 2, 3).split(1, dim=1)
                depth_left, depthright = depth.reshape(-1, 2).split(1, dim=1)
                acc_weight = acc.detach().reshape(-1, 2).min(dim=1, keepdim=True)[0]
                acc_weight[acc_weight < 0.5] = 0
                smooth_loss = self.compute_edge_smooth_loss(rgb_left, rgb_right, depth_left, depthright, acc_weight)
                rgb_left, rgb_right = rgb0.reshape(-1, 2, 3).split(1, dim=1)
                depth_left, depthright = depth0.reshape(-1, 2).split(1, dim=1)
                # acc0 is similar to acc, so reuse acc_weight
                smooth_loss0 = self.compute_edge_smooth_loss(rgb_left, rgb_right, depth_left, depthright, acc_weight)
                others["smooth"] = (smooth_loss0 + smooth_loss).reshape(1, 1).type_as(rgb)

            if self.args.temporal_loss_weight > 0:
                # random generate patches
                stride = np.random.rand() * 2 + 1
                patch_size = 8
                t1 = np.clip(t + np.random.randint(1, 10), 0, self.args.time_len - 1)
                view = torch.tensor([np.random.rand() - 0.5, np.random.rand() - 0.5, 1])
                temporal_loss = self.compute_temporal_loss(t, t1, view, self.args.temporal_loss_patch_num,
                                                           patch_size, stride, rgb)
                others["temporal"] = temporal_loss.reshape(-1, 1).type_as(rgb)

            if self.args.dsmooth_loss_weight > 0 or self.args.uvsmooth_loss_weight > 0:
                ray_num, sample_num, _ = pts.shape
                selected = torch.randint(sample_num, (ray_num, ))
                selected_flat = selected + torch.arange(ray_num) * sample_num
                pts_selected = pts.reshape(-1, 3)[selected_flat]
                uv_smooth, d_smooth = self.compute_uvsmooth_loss(pts_selected, self.mlp_fine, t,
                                                                 ret_uv=self.args.uvsmooth_loss_weight > 0,
                                                                 ret_d=self.args.dsmooth_loss_weight > 0)
                if uv_smooth is not None:
                    others["uv_smooth"] = uv_smooth.reshape(-1, 1)
                if d_smooth is not None:
                    others["d_smooth"] = d_smooth.reshape(-1, 1)

            if self.args.uvprepsmooth_loss_weight > 0:
                ray_num, sample_num, _ = pts.shape
                selected = torch.randint(sample_num, (ray_num, ))
                selected_flat = selected + torch.arange(ray_num) * sample_num
                pts_selected = pts.reshape(-1, 3)[selected_flat]
                uv_smooth, _ = self.compute_uvprepsmooth_loss(pts_selected, self.mlp_fine, t)
                if uv_smooth is not None:
                    others["uvp_smooth"] = uv_smooth.reshape(-1, 1)

            if self.args.gsmooth_loss_weight > 0:
                smth = self.compute_geometry_smooth_loss(t).type_as(rgb)
                others["g_smooth"] = smth.reshape(-1, 1)

            if self.args.kpt_loss_weight > 0 and self.explicit_warp is not None:
                kpt_loss = self.explicit_warp.compute_kpt_loss(t).type_as(rgb)
                others["kpt"] = kpt_loss.reshape(-1, 1)

            # rgb to rgba
            rgb = torch.cat([self.tonemapping(rgb), acc[..., None]], dim=-1)
            rgb0 = torch.cat([self.tonemapping(rgb0), acc0[..., None]], dim=-1)
            return rgb, rgb0, others

        #  evaluation
        else:
            assert poses is not None and intrinsics is not None, "Please specify poses when in the eval model"
            if t is None:
                rgbs, depths = self.render_path_vary_t(H, W, chunk, poses, intrinsics, **kwargs)
            else:
                rgbs, depths = self.render_path_fix_t(H, W, t, chunk, poses, intrinsics, **kwargs)
            return self.tonemapping(rgbs), depths

    def get_texture_map(self, resolution=1024, t=None, views=None):
        with torch.no_grad():
            if isinstance(self.texture_map, TextureMap):
                return (self.rgb_activate(self.texture_map.texture_map),)
            elif isinstance(self.texture_map, (GeneralMLP, TextureMLP, TextureFuse)):
                uv = torch.meshgrid([torch.linspace(-1, 1, resolution), torch.linspace(-1, 1, resolution)])
                uv = [uv[1], uv[0]]  # transpose
                uv = torch.stack(uv, dim=-1).reshape(-1, 2)
                embedded_uv = self.embeduv_fn(uv)
                if t is None:
                    t = 0

                times = torch.tensor(t).float().reshape(-1, 1).expand(len(embedded_uv), -1)
                times = times.type_as(self._example_view_embed)
                ts_embed = self.embedtime_fn_tex(times)

                vx = t / self.args.time_len - 0.5
                if views is None:
                    views = torch.tensor([vx, 0, 1]).float().type_as(self._example_view_embed)
                    views = views / views.norm()
                    views = views.reshape(-1, 3).expand(len(embedded_uv), -1)
                else:
                    views = views.reshape(-1, 3)
                    views = views / views.norm(dim=-1, keepdim=True)
                    views = views.expand(len(embedded_uv), -1)
                vs_embed = self.embeddirs_fn(views)

                embed = torch.cat([embedded_uv.type_as(self._example_view_embed),
                                   vs_embed, ts_embed], dim=-1)
                chunk = 1024 * 4
                outputs = torch.cat([self.texture_map(embed[i: i + chunk]) for i in range(0, len(embed), chunk)],
                                    dim=0)
                if outputs.shape[-1] > 3:
                    final, rgb, rgba = self.texout2rgb(outputs)
                    if self.alpha_type == "add":
                        alphas = rgba[..., -1:]
                        rgba = rgba[..., :-1]
                        rgba = rgba * alphas
                    elif self.alpha_type == "multiply":
                        rgba = torch.sigmoid(rgba * 5)  # multiplier
                    else:
                        raise RuntimeError("unrecognized alpha_type")

                    rgba = rgba.reshape(1, resolution, resolution, 3).permute(0, 3, 1, 2)
                    rgb = rgb.reshape(1, resolution, resolution, 3).permute(0, 3, 1, 2)
                    final = final.reshape(1, resolution, resolution, 3).permute(0, 3, 1, 2)
                    texture = (rgba, rgb, final)
                else:
                    texture = self.rgb_activate(outputs[:, :3]) \
                        .reshape(1, resolution, resolution, 3) \
                        .permute(0, 3, 1, 2)
                    texture = (texture,)
                return texture
            else:
                raise RuntimeError()

    def compute_sparsity_loss(self, weight):
        # weight is of shape (B, SampleNum)
        if self.args.sparsity_type == 'none':
            weight = weight[:, :-1]  # remove the last sample point cause it's usually ambiguous
            return torch.tensor(0).type_as(weight)
        elif self.args.sparsity_type == 'l1':
            weight = weight[:, :-1]  # remove the last sample point cause it's usually ambiguous
            return weight.abs().mean()
        elif self.args.sparsity_type == 'l1/l2':
            weight = weight[:, :-1]  # remove the last sample point cause it's usually ambiguous
            norm = weight.abs().sum(dim=-1) / weight.norm(dim=-1).clamp_min(0.001)
            return norm.mean()
        elif self.args.sparsity_type == 'entropy':
            entropy = -(weight * torch.log(weight.clamp_min(1e-10))).sum(dim=-1)
            return entropy.mean()
        else:
            raise RuntimeError(f"Unrecognized error {self.args.sparsity_type}")

    def compute_cycle_loss(self, pts_in, pts_out, t, uv2xyz, weight, netchunk=1024 * 64):
        # for cycle loss
        pts_in = pts_in.reshape(-1, 3)
        pts_out = pts_out.reshape(-1, 3)
        input_time = t * torch.ones_like(pts_in[:, :1])
        embedded4cycle = self.embed_fn_cycle(pts_out)
        embedded_time4cycle = self.embedtime_fn_cycle(input_time)
        embedded4cycle = torch.cat([embedded4cycle, embedded_time4cycle], dim=-1)
        xyz_remap = torch.cat([uv2xyz(embedded4cycle[i:i + netchunk])
                               for i in range(0, embedded4cycle.shape[0], netchunk)], 0)
        xyz_remap = xyz_remap - pts_in

        diff = (xyz_remap ** 2).sum(dim=-1).reshape(weight.shape)
        diff = (weight * diff).sum(dim=-1)
        return diff.mean()

    def compute_uvprepsmooth_loss(self, pts, mlp, t, ret_uv=False, ret_d=False):
        pts.requires_grad = True
        latent_t = t * torch.ones_like(pts[..., :1])
        density_raw, uv, _, _, _ = self.mlpforward(pts, None, latent_t, mlp, None)
        d_output = torch.ones_like(uv[:, 0:1], requires_grad=False)

        # read the uv weight files
        weight = torchf.grid_sample(self.uv_weight_map, uv[..., :2].reshape(1, 1, -1, 2),
                                    mode='bilinear', padding_mode="border")
        weight = weight.reshape(-1)

        # read the uv weight files
        gu = torch.autograd.grad(
            outputs=uv[:, 0:1],
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,  # graph of the derivative will be constructed, allowing to compute
            # higher order derivative products
            retain_graph=True,
            only_inputs=True)[0]
        gv = torch.autograd.grad(
            outputs=uv[:, 1:2],
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,  # graph of the derivative will be constructed, allowing to compute
            # higher order derivative products
            retain_graph=True,
            only_inputs=True)[0]
        gd = torch.autograd.grad(
            outputs=density_raw,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,  # graph of the derivative will be constructed, allowing to compute
            # higher order derivative products
            retain_graph=True,
            only_inputs=True)[0].detach()

        gd_2 = (gd ** 2).sum(-1).clamp_min(1e-8)
        gu_proj2gd = ((gu * gd).sum(-1) / gd_2)[..., None] * gd
        gv_proj2gd = ((gv * gd).sum(-1) / gd_2)[..., None] * gd
        gu_T, gv_T = gu - gu_proj2gd, gv - gv_proj2gd

        denorm = (gu_T.norm(dim=-1) * gv_T.norm(dim=-1)).clamp_min(1e-8)
        smoothuv = ((gu_T * gu_T).sum(dim=-1) / denorm) ** 2
        smoothuv = (smoothuv * weight).mean()
        return smoothuv, None

    def compute_uvsmooth_loss(self, pts, mlp, t, ret_uv=False, ret_d=False):
        pts.requires_grad = True
        latent_t = t * torch.ones_like(pts[..., :1])
        density_raw, uv, _, _, _ = self.mlpforward(pts, None, latent_t, mlp, None)
        d_output = torch.ones_like(uv[:, 0:1], requires_grad=False)

        if ret_uv:
            # read the uv weight files
            weight = torchf.grid_sample(self.uv_weight_map, uv[..., :2].reshape(1, 1, -1, 2),
                                        mode='bilinear', padding_mode="border")
            weight = weight.reshape(-1)

            # read the uv weight files
            gradients_x = torch.autograd.grad(
                outputs=uv[:, 0:1],
                inputs=pts,
                grad_outputs=d_output,
                create_graph=True,  # graph of the derivative will be constructed, allowing to compute
                # higher order derivative products
                retain_graph=True,
                only_inputs=True)[0]
            gradients_y = torch.autograd.grad(
                outputs=uv[:, 1:2],
                inputs=pts,
                grad_outputs=d_output,
                create_graph=True,  # graph of the derivative will be constructed, allowing to compute
                # higher order derivative products
                retain_graph=True,
                only_inputs=True)[0]

            denorm = (gradients_x.norm(dim=-1) * gradients_y.norm(dim=-1)).clamp_min(1e-6)
            smoothuv = ((gradients_x * gradients_y).sum(dim=-1) / denorm) ** 2
            smoothuv = (smoothuv * weight).mean()
        else:
            smoothuv = None

        if ret_d:
            assert ret_uv
            gradients_d = torch.autograd.grad(
                outputs=density_raw,
                inputs=pts,
                grad_outputs=d_output,
                create_graph=True,  # graph of the derivative will be constructed, allowing to compute
                # higher order derivative products
                retain_graph=True,
                only_inputs=True)[0].detach()
            denorm_d = gradients_d.norm(dim=-1)
            denorm_ud = (denorm_d * gradients_x.norm(dim=-1)).clamp_min(1e-6)
            smooth_ud = ((gradients_d * gradients_x).sum(dim=-1) / denorm_ud) ** 2
            denorm_vd = (denorm_d * gradients_y.norm(dim=-1)).clamp_min(1e-6)
            smooth_vd = ((gradients_d * gradients_y).sum(dim=-1) / denorm_vd) ** 2
            smoothd = smooth_ud.mean() + smooth_vd.mean()
        else:
            smoothd = None
        return smoothuv, smoothd

    def compute_alpha_loss(self, alphas, weight):
        if self.texture_type != "fuse" or alphas is None:
            return torch.tensor(0.)

        if self.alpha_type == "add":
            alphas = self.rgb_activate(alphas[..., -1])
            loss = (alphas - 0.004).abs().mean()
        elif self.alpha_type == "multiply":
            mrgb = alphas[..., :3].abs() * weight[..., None]
            loss = mrgb.mean() * 3
        else:
            raise RuntimeError(f"Unrecognized alpha type {self.alpha_type}")
        return loss

    def compute_temporal_loss(self, t0, t1, views, patch_num, patch_size, stride, tensorformat):
        linspace = torch.linspace(0, 2 * patch_size * stride / self.args.texture_map_resolution, patch_size)
        meshgrid = torch.meshgrid([linspace, linspace])
        meshgrid = torch.stack(meshgrid, dim=-1).reshape(-1, 2)
        coord_min = -1
        coord_max = 1 - 2 * patch_size * stride / self.args.texture_map_resolution
        dxdy = torch.rand(patch_num, 2) * (coord_max - coord_min) + coord_min
        uv = meshgrid[None, ...] + dxdy[:, None]  # P x 64 x 2
        uv_flat = uv.reshape(-1, 2)
        uv_embed = self.embeduv_fn(uv_flat)

        views = views / views.norm()
        views = views.reshape(-1, 3).expand(len(uv_embed), -1)
        vs_embed = self.embeddirs_fn(views)

        t0 = torch.tensor(t0).type_as(tensorformat).reshape(-1, 1).expand(len(uv_embed), -1)
        t1 = torch.tensor(t1).type_as(tensorformat).reshape(-1, 1).expand(len(uv_embed), -1)
        ts_embed0 = self.embedtime_fn_tex(t0)
        ts_embed1 = self.embedtime_fn_tex(t1)

        embed0 = torch.cat([uv_embed, vs_embed, ts_embed0], dim=-1)
        embed1 = torch.cat([uv_embed, vs_embed, ts_embed1], dim=-1)
        output0 = self.texture_map(embed0).reshape(patch_num, patch_size * patch_size, -1)
        output1 = self.texture_map(embed1).reshape(patch_num, patch_size * patch_size, -1)

        rgb0, _, _ = self.texout2rgb(output0)
        rgb1, _, _ = self.texout2rgb(output1)
        # compute ssim loss
        weights = torch.tensor([0.2126, 0.7152, 0.0722]).reshape(1, 1, 3, 1).type_as(rgb0)
        c2, c3 = 0.0009, 0.00045
        luma0 = (rgb0[..., None, :] @ weights)[..., 0, 0]
        luma1 = (rgb1[..., None, :] @ weights)[..., 0, 0]
        mean0, mean1 = luma0.mean(dim=-1, keepdim=True), luma1.mean(dim=-1, keepdim=True)
        sigma0 = torch.std(luma0, dim=-1)
        sigma1 = torch.std(luma1, dim=-1)
        sigma01 = ((luma0 - mean0) * (luma1 - mean1)).sum(dim=-1) / luma0.shape[-1]
        sig0sig1 = sigma0 * sigma1
        sim = (2 * sig0sig1 + c2) * (sigma01.abs() + c3) / ((sigma0 ** 2 + sigma1 ** 2 + c2) * (sig0sig1 + c3))
        return - sim.mean()

    def compute_edge_smooth_loss(self, rgb0, rgb1, disp0, disp1,
                                 additional_weight=None, rgb_slope=1, disp_margin=0.01):
        rgb_diff = ((rgb0 - rgb1).abs()).sum(dim=-1)
        weight = - (rgb_diff * rgb_slope).clamp_max(1) + 1
        disp_diff = ((disp0 - disp1).abs() - disp_margin).clamp_min(0)
        smooth = weight * disp_diff
        if additional_weight is not None:
            smooth = (smooth * additional_weight).sum() / additional_weight.sum().clamp_min(1)
        else:
            smooth = smooth.mean()
        return smooth.mean()

    def compute_geometry_smooth_loss(self, t):
        loss = torch.tensor(0.)

        if self.args.gsmooth_loss_type == 'o1':
            t0 = max(t - 3, 1)
            t1 = min(t + 3, self.args.time_len)
            if hasattr(self.explicit_warp, "transform"):
                transform = self.explicit_warp.transform
                loss = (transform[t0:t1] - transform[t0-1:t1-1]).abs().mean() + loss.type_as(transform)
            if hasattr(self.explicit_warp, "kpt3d"):
                kpts = self.explicit_warp.kpt3d
                loss = (kpts[t0:t1] - kpts[t0-1:t1-1]).abs().mean() + loss.type_as(kpts)
        elif self.args.gsmooth_loss_type == 'o2':
            t0 = max(t - 8, 1)
            t1 = min(t + 8, self.args.time_len - 1)
            if hasattr(self.explicit_warp, "transform"):
                tran0 = self.explicit_warp.transform[t0-1:t1-1]
                tran1 = self.explicit_warp.transform[t0:t1]
                tran2 = self.explicit_warp.transform[t0+1:t1+1]
                loss = (2 * tran1 - tran0 - tran2).abs().mean() * 0.3 + loss.type_as(tran0)
            if hasattr(self.explicit_warp, "kpt3d"):
                kpts0 = self.explicit_warp.kpt3d[t0-1:t1-1]
                kpts1 = self.explicit_warp.kpt3d[t0:t1]
                kpts2 = self.explicit_warp.kpt3d[t0+1:t1+1]
                loss = (2 * kpts1 - kpts0 - kpts2).abs().mean() + loss.type_as(kpts0)
            if hasattr(self.explicit_warp, "kpt3d_bias_radius") and len(self.explicit_warp.kpt3d_bias_radius) > 1:
                rad0 = self.explicit_warp.kpt3d_bias_radius[t0-1:t1-1]
                rad1 = self.explicit_warp.kpt3d_bias_radius[t0:t1]
                rad2 = self.explicit_warp.kpt3d_bias_radius[t0+1:t1+1]
                loss = (2 * rad1 - rad0 - rad2).abs().mean() + loss.type_as(rad0)
        else:
            raise RuntimeError(f"{self.args.gsmooth_loss_type} not recognized")
        return loss

    def texout2rgb(self, texture_output):
        rgb0 = self.rgb_activate(texture_output[..., :3])  # [N_rays, N_samples, 3]
        if texture_output.shape[-1] < 5:
            rgb = rgb0
            return rgb, None, None
        if self.alpha_type == "add":
            rgba = self.rgb_activate(texture_output[..., 3:7])
            rgb_, a_ = rgba.split([3, 1], dim=-1)
            rgb = rgb_ * a_ + rgb0 * (- a_ + 1)
            rgb_ = rgba
        elif self.alpha_type == "multiply":
            rgb_ = texture_output[..., 3:6]
            rgb = (torch.exp(rgb_) * rgb0).clamp(0, 1)
        else:
            raise RuntimeError(f"Unrecognized alpha_type {self.alpha_type}")
        return rgb, rgb0, rgb_

    def raw2outputs_old(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        dists = (z_vals[..., 1:] - z_vals[..., :-1]).abs()
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb, _, _ = self.texout2rgb(raw)

        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., -1].shape) * raw_noise_std

        sigma = self.sigma_activate(raw[..., -1] + noise)
        alpha = - torch.exp(- sigma * dists) + 1.  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        disp_map = torch.reciprocal((depth_map / acc_map).clamp_min(1e-10))

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None]) * (28 / 255)

        return rgb_map, disp_map, acc_map, weights, depth_map

    def render_rays(self,
                    ray_batch,
                    ray_infos,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False,
                    use_viewdirs=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 6 else None
        near, far, times = torch.split(ray_infos.reshape(-1, 3), 1, dim=-1)
        miss_mask = far <= near
        near[miss_mask] = 2
        far[miss_mask] = 4  # really don't know why leads to the nan
        intersec_mask = torch.logical_not(miss_mask).float()

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o) * perturb

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw, uv, pts_warp, pts_out, alphas = self.mlpforward(pts, viewdirs, times,
                                                             self.mlp_coarse, self.texture_map_coarse)
        if self.args.render_keypoints:
            k_sigma, k_color = self.explicit_warp.render(pts, times)
            k_sigma = k_sigma[..., None] / 2
            old_sigma = raw[..., -1:]
            old_color = raw[..., :3]

            k_weight = (k_sigma > old_sigma).type(torch.float32)
            new_sigma = k_sigma + old_sigma
            new_color = k_weight * k_color + (1 - k_weight) * old_color
            raw = torch.cat([new_color, raw[..., 3:-1], new_sigma], dim=-1)

        rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs_old(
            raw, z_vals, rays_d, raw_noise_std,
            white_bkgd, pytest=pytest)
        rgb_map = rgb_map * intersec_mask
        acc_map = acc_map * intersec_mask[..., 0]
        weights = weights * intersec_mask
        depth_map = depth_map * intersec_mask[..., 0]

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0 = rgb_map, depth_map, acc_map, density_map
            uv_0 = uv
            pts_0 = pts
            weights_0 = weights
            pts_warp_0 = pts_warp
            pts_out_0 = pts_out
            alphas0 = alphas

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()
            if self.args.N_samples_fine < z_vals.shape[-1]:
                choice = np.random.choice(z_vals.shape[-1], self.args.N_samples_fine, replace=False)
                z_vals = z_vals[:, choice]
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            raw, uv, pts_warp, pts_out, alphas = self.mlpforward(pts, viewdirs, times,
                                                                 self.mlp_fine, self.texture_map)
            if self.args.render_keypoints:
                k_sigma, k_color = self.explicit_warp.render(pts, times)
                k_color = torch.log(k_color/(1-k_color))  # undo sigmoid
                k_sigma = k_sigma[..., None]
                old_sigma = raw[..., -1:] / 2
                old_color = raw[..., :3]
                old_color1 = raw[..., 3:-1]

                k_weight = (k_sigma > old_sigma).type(torch.float32)
                new_sigma = k_sigma + old_sigma
                new_color = k_weight * k_color + (1 - k_weight) * old_color
                new_color1 = (1 - k_weight) * old_color1
                raw = torch.cat([new_color, new_color1, new_sigma], dim=-1)

            rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs_old(raw, z_vals, rays_d, raw_noise_std,
                                                                                     white_bkgd, pytest=pytest)
            rgb_map = rgb_map * intersec_mask
            acc_map = acc_map * intersec_mask[..., 0]
            weights = weights * intersec_mask
            depth_map = depth_map * intersec_mask[..., 0]

        ret = {'rgb_map': rgb_map,
               'depth_map': depth_map,
               'acc_map': acc_map,
               'density_map': density_map,
               'uv': uv,
               'pts': pts,
               'weight_map': weights}
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
            ret['uv0'] = uv_0
            ret['pts0'] = pts_0
            ret['weight0'] = weights_0
        if pts_warp is not None:
            ret['pts_warp'] = pts_warp
        if pts_warp_0 is not None:
            ret['pts_warp0'] = pts_warp_0
        if pts_out is not None:
            ret['pts_out'] = pts_out
        if pts_out_0 is not None:
            ret['pts_out0'] = pts_out_0
        if alphas is not None:
            ret['alpha'] = alphas
        if alphas0 is not None:
            ret['alpha0'] = alphas0

        for k in ret:
            if "density" in k:
                continue
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def render(self, H, W, rays=None, t=None, box=None,
               ndc=False, chunk=1024, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays

        # provide ray directions as inputs
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays_tensor(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        # if ndc:
        #     # for forward facing scenes
        #     rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        t = t * torch.ones_like(rays_d[..., :1])
        # near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        # find near far by intersection
        with torch.no_grad():
            box = torch.tensor(box).type_as(rays_o)
            rays_d_denorm = torch.reciprocal(rays_d).clamp(-1e10, 1e10)
            intersec1 = (box[0:1] - rays_o) * rays_d_denorm
            intersec2 = (box[1:2] - rays_o) * rays_d_denorm
            near = torch.min(intersec1, intersec2).max(dim=-1, keepdim=True)[0].clamp_min(0)
            far = torch.max(intersec1, intersec2).min(dim=-1, keepdim=True)[0]
            far = torch.max(near, far)
        # print(f"near min {near.min()}, near max {near.max()}, far min {far.min()}, far max {far.max()}")
        rays = torch.cat([rays_o, rays_d, viewdirs], -1)
        rays_info = torch.cat([near, far, t], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], rays_info[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def force_load_texture_map(self, path, isfull=False, force_map=False):
        if self.texture_type == "mlp":
            self.texture_map = TextureMap(
                resolution=self.args.texture_map_resolution,
                face_roi=self.args.uv_map_face_roi,
                output_ch=self.args.texture_map_channel,
                activate=activate[self.args.rgb_activate],  # for undo the activate while loading
                grad_multiply=self.args.texture_map_gradient_multiply
            )
            self.texture_map_coarse = self.texture_map
            print(f"Load texture map from {path}")
            self.texture_map.load(path, isfull)

        elif self.texture_type == "fuse":
            self.texture_map.promote_texture(mlp2map=False)
            self.texture_map.map.load(path, isfull)
            if force_map:
                self.texture_map = self.texture_map.map
        elif self.texture_type == "map":
            self.texture_map.load(path, isfull)
        else:
            raise RuntimeError(f"Unrecongnized texture_type {self.texture_type}")

    def force_load_geometry_map(self, path, isfull=False):
        self.density_offset = TextureMap(
            resolution=self.args.texture_map_resolution,
            face_roi=self.args.uv_map_face_roi,
            output_ch=1,
            activate="geometry"  # for undo the activate while loading
        )
        self.density_offset.load(path, isfull)
        print("Successfully load density offset")

    def render_path_fix_t(self, H, W, t, chunk, render_poses, intrinsics, render_kwargs):
        rgbs = []
        depths = []

        # print("rendering frame: ", end='')
        for i, (c2w, K) in enumerate(zip(render_poses, intrinsics)):
            print(f"{i}", end=' ', flush=True)
            rays = get_rays_tensor(H, W, K, c2w)
            rays = torch.stack(rays, dim=0)
            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **render_kwargs)
            if self.args.render_rgba:
                rgb = torch.cat([rgb, acc[..., None]], dim=-1)

            rgbs.append(rgb)
            depths.append(depth)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_path_vary_t(self, H, W, chunk, render_poses, intrinsics, render_kwargs):
        rgbs = []
        depths = []

        print("rendering frame: ", end='')
        for i in range(render_poses.shape[0]):
            print(f"{i}", end=' ', flush=True)
            c2w = render_poses[i]
            K = intrinsics[i]
            t = i % self.args.time_len
            rays = get_rays_tensor(H, W, K, c2w)
            rays = torch.stack(rays, dim=0)

            rgb, depth, acc, extras = self.render(H, W, rays, t, chunk=chunk, **render_kwargs)

            if self.args.render_rgba:
                rgb = torch.cat([rgb, acc[..., None]], dim=-1)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths
