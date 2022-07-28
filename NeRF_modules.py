import torch
import torch.nn as nn
import imageio
import cv2
import torch.nn.functional as torchf
import numpy as np
from NGP_modules import HashEmbedder


# Positional encoding (section 5.1)
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)


class EmbedderTime(Embedder):
    def __init__(self, **kwargs):
        assert kwargs['input_dims'] == 1
        super().__init__(**kwargs)
        self.dict_len = kwargs["dict_len"]

    def foward(self, x):
        super(EmbedderTime, self).foward(x / self.dict_len)


# Positional encoding (section 5.1)
class EmbedderWindowed(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim
        self.freq_weight = np.ones(N_freqs)
        self.window_start = self.kwargs['window_start']
        self.window_size = self.kwargs['window_end'] - self.window_start
        self.update_activate_freq(1e15)

    def update_activate_freq(self, step):
        alpha = (step - self.window_start) / self.window_size
        alpha = max(min(alpha, 1), 0)
        alpha = alpha * len(self.freq_bands)
        freq_bands_idx = np.arange(len(self.freq_bands))
        self.freq_weight = (1 - np.cos(np.pi * np.clip(alpha - freq_bands_idx, 0, 1))) / 2

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq, freq_w in zip(self.freq_bands, self.freq_weight):
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq) * freq_w)
        return torch.cat(outputs, -1)


class EmbedderTimeWindowed(EmbedderWindowed):
    def __init__(self, **kwargs):
        assert kwargs['input_dims'] == 1
        super().__init__(**kwargs)
        self.dict_len = kwargs["dict_len"]

    def foward(self, x):
        super(self).foward(x / self.dict_len)


class DictEmbedder(nn.Module):
    def __init__(self, latent_size, dict_len):
        super(DictEmbedder, self).__init__()
        latent_tdirs = torch.zeros(dict_len, latent_size)
        self.register_parameter("latent_tdirs", nn.Parameter(latent_tdirs, requires_grad=True))

    def forward(self, x):
        x = x.type(torch.long).squeeze(-1)
        return self.latent_tdirs[x]


class DictEmbedderWindowed(nn.Module):
    def __init__(self, latent_size, dict_len, end_step):
        super(DictEmbedderWindowed, self).__init__()
        latent_tdirs = torch.zeros(dict_len, latent_size)
        self.register_parameter("latent_tdirs", nn.Parameter(latent_tdirs, requires_grad=True))
        self.end_step = end_step
        self.step = 0
        print(f"Setting dict embedder's end_step to {self.end_step}")

    def update_activate_freq(self, step):
        self.step = step

    def forward(self, x):
        x = x.type(torch.long).squeeze(-1)
        embed = self.latent_tdirs[x]
        if self.step < self.end_step:
            embed = embed.detach()
        return embed


def get_embedder(multires, embed_type='pe', input_dim=3,
                 window_start=0, window_end=-1,  # when end>0 means windowed embedder
                 dict_len=-1, latent_size=-1,  # when >0 means time embedder else general embedder
                 log2_hash_size=19, finest_resolution=1024  # args for hashembedder
                 ):
    if (latent_size < 0 and embed_type == "latent") \
            or (multires < 0 and embed_type == "pe")\
            or (multires < 0 and embed_type == "hash"):
        return lambda x: torch.ones_like(x[..., :0]), 0
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    if embed_type == "pe":
        # if dict_len > 0:  # time embedder
        #     embed_kwargs["dict_len"] = dict_len
        #     embed_kwargs["input_dims"] = 1
        #     if window_end <= 0:
        #         embedder_obj = EmbedderTime(**embed_kwargs)
        #     else:
        #         embed_kwargs["window_start"] = window_start
        #         embed_kwargs["window_end"] = window_end
        #         embedder_obj = EmbedderTimeWindowed(**embed_kwargs)
        # else:  # other embedder
        if window_end <= 0:
            embedder_obj = Embedder(**embed_kwargs)
        else:
            embed_kwargs["window_start"] = window_start
            embed_kwargs["window_end"] = window_end
            embedder_obj = EmbedderWindowed(**embed_kwargs)
        return embedder_obj, embedder_obj.out_dim

    elif embed_type == "none":
        return lambda x: x, input_dim

    elif embed_type == "latent":
        if window_end <= 0:
            return DictEmbedder(latent_size, dict_len), latent_size
        else:
            return DictEmbedderWindowed(latent_size, dict_len, window_end), latent_size

    elif embed_type == "hash":
        embed = HashEmbedder(n_indim=input_dim,
                             log2_hashmap_size=log2_hash_size,
                             finest_resolution=2 ** multires)
        return embed, embed.out_dim

    else:
        raise RuntimeError(f"Unrecognized embedder type {embed_type}")


# Model
class NeRFmlp(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_latent_t=0,
                 output_ch=4, skips=[4], use_viewdirs=False):
        """
            input_ch_latent_t: set to 0 to disable latent_t
        """
        super(NeRFmlp, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_latent_t = input_ch_latent_t
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch + input_ch_latent_t, W)]
            + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)] + [nn.Linear(W // 2, W // 2) for i in range(D // 2)])

        if use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views, input_latent_t = torch.split(x, [self.input_ch, self.input_ch_views,
                                                                 self.input_ch_latent_t], dim=-1)
        h = torch.cat([input_pts, input_latent_t], -1)

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# Model
class GeneralMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_view=0, input_ch_time=0,
                 view_layer_idx=0, time_layer_idx=0,
                 output_ch=3, skips=[4]):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.input_ch_views = input_ch_view
        self.view_layer_idx = [view_layer_idx] if isinstance(view_layer_idx, int) else view_layer_idx
        self.time_layer_idx = [time_layer_idx] if isinstance(time_layer_idx, int) else time_layer_idx
        self.view_layer_idx = [i if i >= 0 else D + i for i in self.view_layer_idx]
        self.time_layer_idx = [i if i >= 0 else D + i for i in self.time_layer_idx]
        self.skips = skips

        assert all([i < D - 1 for i in self.time_layer_idx])
        assert all([i < D - 1 for i in self.view_layer_idx])
        assert all([i < D - 1 for i in self.skips])

        layers = []
        for i in range(D):
            cnl_in = self.input_ch if i == 0 else W

            if i in skips:
                cnl_in += self.input_ch
            if i in self.view_layer_idx:
                cnl_in += self.input_ch_views
            if i in self.time_layer_idx:
                cnl_in += self.input_ch_time

            cnl_out = output_ch if i == D - 1 else W
            layers.append(nn.Linear(cnl_in, cnl_out))
        self.mlp = nn.ModuleList(layers)

    def forward(self, x):
        input_pts, input_views, input_time = torch.split(x, [self.input_ch, self.input_ch_views,
                                                             self.input_ch_time], dim=-1)
        h = input_pts
        for i, layer in enumerate(self.mlp[:-1]):
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)
            if i in self.view_layer_idx:
                h = torch.cat([h, input_views], -1)
            if i in self.time_layer_idx:
                h = torch.cat([h, input_time], -1)
            h = torch.relu(layer(h))
        return self.mlp[-1](h)


# texture map
class TextureMap(nn.Module):
    def __init__(self, resolution, face_roi, output_ch=3, activate=None, grad_multiply=1.):
        super().__init__()
        self.resolution = resolution
        self.cnl = output_ch
        self.face_roi = face_roi
        self.activate = activate if activate is not None else lambda x: x

        texture_map = torch.zeros(1, output_ch, resolution, resolution)
        self.register_parameter("texture_map", nn.Parameter(texture_map, requires_grad=True))

        def increase_grad_hook(module, grad_in, grad_out):
            return (grad_in[0] * grad_multiply,)

        if grad_multiply > 1:
            self.register_backward_hook(increase_grad_hook)

    def load(self, path, isfull=False):
        initial_texture_map = imageio.imread(path)
        if isfull:
            size_face = self.resolution
            start_ = 0
        else:
            size_face = int(self.face_roi * self.resolution)
            start_ = (self.resolution - size_face) // 2
        print(f"face_roi = {self.face_roi}")
        print(f"Resizing the texture map from {initial_texture_map.shape} to {(size_face, size_face)}")
        initial_texture_map = cv2.resize(initial_texture_map, (size_face, size_face),
                                         interpolation=cv2.INTER_LINEAR)
        initial_texture_map = initial_texture_map / 255
        if initial_texture_map.shape[-1] == 4:
            initial_texture_map = initial_texture_map[..., :3] * initial_texture_map[..., 3:4]
        initial_texture_map = (torch.tensor(initial_texture_map).permute(2, 0, 1)).type_as(self.texture_map)
        # redo sigmoid if needed
        if self.activate == torch.sigmoid:
            initial_texture_map = torch.log(initial_texture_map / (- initial_texture_map + 1.)).clamp(-10, 10)
        elif self.activate == "geometry":
            initial_texture_map = (initial_texture_map - 128 / 255) * 20000

        initial_texture_cnl = initial_texture_map.shape[0]
        with torch.no_grad():
            if self.cnl == initial_texture_cnl:
                self.texture_map.detach()[0, :, start_:start_ + size_face, start_:start_ + size_face] \
                    = initial_texture_map
            elif self.cnl > initial_texture_cnl:
                print(f"Warning: the texture map has {self.cnl} channels, but the image has {initial_texture_cnl} cnls"
                      f" will only load as the first {initial_texture_cnl} channel")
                self.texture_map.detach()[0, :initial_texture_cnl, start_:start_ + size_face, start_:start_ + size_face] \
                    = initial_texture_map
            else:  # self.cnl < initial_texture_cnl
                print(f"Warning: the texture map has {self.cnl} channels, but the image has {initial_texture_cnl} cnls"
                      f" will load the first {self.cnl} channel")
                self.texture_map.detach()[0, :, start_:start_ + size_face, start_:start_ + size_face] \
                    = initial_texture_map[:self.cnl]
        return

    def forward(self, x):
        shape_ori = x.shape[:-1]
        rgb = torchf.grid_sample(self.texture_map, x[..., :2].reshape(1, 1, -1, 2),
                                 mode='bilinear', padding_mode="zeros")
        rgb = rgb.permute(0, 2, 3, 1).reshape(*shape_ori, -1)
        return rgb


# Model
class TextureMLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=0, input_ch_latent_t=0,
                 output_ch=4, skips=(4,)):
        """
            input_ch_latent_t: set to 0 to disable latent_t
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_latent_t = input_ch_latent_t
        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])

        ### Implementation according to the paper
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + input_ch_latent_t + W, W)] + [nn.Linear(W, W) for i in range(2)])

        if input_ch_views > 0:
            self.alpha_linear = nn.Linear(W, 1)
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W, output_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views, input_latent_t = torch.split(x, [self.input_ch, self.input_ch_views,
                                                                 self.input_ch_latent_t], dim=-1)
        h = input_pts
        # TODO
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = torch.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.input_ch_views > 0:
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views, input_latent_t], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = torch.relu(h)

            outputs = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)

        return outputs


# Model
class TextureFuse(nn.Module):
    def __init__(self, *, uv_embedder, D=8, W=256, input_ch=3,
                 input_ch_view=0, input_ch_time=0,
                 view_layer_idx=0, time_layer_idx=0, skips=(3,),
                 resolution=1024, face_roi=0.8, output_ch=3, activate=None,
                 texture_map_gradient_multiply=1.):
        """
            input_ch_latent_t: set to 0 to disable latent_t
        """
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_view
        self.input_ch_latent_t = input_ch_time
        self.skips = skips
        self.resolution = resolution
        self.cnl = output_ch
        self.face_roi = face_roi
        self.activate = activate if activate is not None else lambda x: x
        self.uv_embed_fn = uv_embedder
        self.texture_map_gradient_multiply = texture_map_gradient_multiply

        self.mlp = GeneralMLP(D, W, input_ch, input_ch_view, input_ch_time,
                              view_layer_idx=view_layer_idx, time_layer_idx=time_layer_idx, skips=skips,
                              output_ch=output_ch + 1)

        # will only use map_mlp or map_tex
        self.map = GeneralMLP(D, W, input_ch, 0, 0, skips=skips,
                              output_ch=output_ch)
        self.map_overlay = None

    def promote_texture(self, mlp2map=True):
        if isinstance(self.map, TextureMap):
            print("TextureFuse::Warning!!! the map is alread a texture map, "
                  "which shouldn't happen, will do nothing but return")
            return

        if not mlp2map:
            print("TextureFuse::Warning!!! mlp2map is set to False, which shouldn't happened usually")
            self.map = TextureMap(self.resolution, self.face_roi, self.cnl, self.activate,
                                  self.texture_map_gradient_multiply)
            return

        with torch.no_grad():
            uv = torch.meshgrid([torch.linspace(-1, 1, self.resolution), torch.linspace(-1, 1, self.resolution)])
            uv = [uv[1], uv[0]]  # transpose
            uv = torch.stack(uv, dim=-1).reshape(-1, 2)
            embed = self.uv_embed_fn(uv)
            chunk = 1024 * 4
            outputs = torch.cat([self.map(embed[i: i + chunk]) for i in range(0, len(embed), chunk)],
                                dim=0)
            texture = outputs.reshape(1, self.resolution, self.resolution, 3).permute(0, 3, 1, 2)
            self.map = TextureMap(self.resolution, self.face_roi, self.cnl, self.activate,
                                  self.texture_map_gradient_multiply)
            self.map.texture_map.copy_(texture)
        print("TextureFuse::the first layer now converted to explicit texture map")

    def forward(self, x):
        map_rgb = self.map(x[..., :self.input_ch])
        mlp_rgba = self.mlp(x)

        # if self.map_overlay is not None:

        return torch.cat([map_rgb, mlp_rgba], dim=-1)
