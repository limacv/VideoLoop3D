import torch
import torch.nn as nn
import torch.nn.functional as torchf
import numpy as np

primes_all = torch.tensor([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737])


@torch.no_grad()
def hash_coord(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    dim = coords.shape[-1]
    global primes_all
    primes = primes_all[:dim].reshape(1, 1, dim).type_as(coords)
    xor_result0 = coords * primes
    xor_result = torch.bitwise_xor(xor_result0[..., 0], xor_result0[..., 1])
    if dim >= 3:
        xor_result = torch.bitwise_xor(xor_result, xor_result0[..., 2])
    if dim >= 4:
        xor_result = torch.bitwise_xor(xor_result, xor_result0[..., 3])
    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


def bilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: B x 2
    voxel_min_vertex: B x 2
    voxel_max_vertex: B x 2
    voxel_embedds: B x 4 x 2
    '''
    # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
    weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 2

    # step 1
    c0 = voxel_embedds[:, 0] * (1 - weights[:, 0:1]) + voxel_embedds[:, 2] * weights[:, 0:1]
    c1 = voxel_embedds[:, 1] * (1 - weights[:, 0:1]) + voxel_embedds[:, 3] * weights[:, 0:1]

    # step 2
    c = c0 * (1 - weights[:, 1:2]) + c1 * weights[:, 1:2]

    return c


def trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: B x 3
    voxel_min_vertex: B x 3
    voxel_max_vertex: B x 3
    voxel_embedds: B x 8 x 2
    '''
    # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
    weights = (x - voxel_min_vertex) * 2 / (voxel_max_vertex - voxel_min_vertex) - 1  # B x 3
    c = torchf.grid_sample(voxel_embedds.reshape(-1, 2, 2, 2, 2).permute(0, 4, 3, 2, 1),
                           weights.reshape(-1, 1, 1, 1, 3),
                           padding_mode='border',
                           align_corners=True).squeeze()
    return c


def quadlinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: B x 3
    voxel_min_vertex: B x 3
    voxel_max_vertex: B x 3
    voxel_embedds: B x 8 x 2
    '''
    # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
    weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

    # step 1
    # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
    c000 = voxel_embedds[:, 0] * (1 - weights[:, 0:1]) + voxel_embedds[:, 8] * weights[:, 0:1]
    c001 = voxel_embedds[:, 1] * (1 - weights[:, 0:1]) + voxel_embedds[:, 9] * weights[:, 0:1]
    c010 = voxel_embedds[:, 2] * (1 - weights[:, 0:1]) + voxel_embedds[:, 10] * weights[:, 0:1]
    c011 = voxel_embedds[:, 3] * (1 - weights[:, 0:1]) + voxel_embedds[:, 11] * weights[:, 0:1]
    c100 = voxel_embedds[:, 4] * (1 - weights[:, 0:1]) + voxel_embedds[:, 12] * weights[:, 0:1]
    c101 = voxel_embedds[:, 5] * (1 - weights[:, 0:1]) + voxel_embedds[:, 13] * weights[:, 0:1]
    c110 = voxel_embedds[:, 6] * (1 - weights[:, 0:1]) + voxel_embedds[:, 14] * weights[:, 0:1]
    c111 = voxel_embedds[:, 7] * (1 - weights[:, 0:1]) + voxel_embedds[:, 15] * weights[:, 0:1]

    c00 = c000 * (1 - weights[:, 1:2]) + c100 * weights[:, 1:2]
    c01 = c001 * (1 - weights[:, 1:2]) + c101 * weights[:, 1:2]
    c10 = c010 * (1 - weights[:, 1:2]) + c110 * weights[:, 1:2]
    c11 = c011 * (1 - weights[:, 1:2]) + c111 * weights[:, 1:2]

    # step 2
    c0 = c00 * (1 - weights[:, 2:3]) + c10 * weights[:, 2:3]
    c1 = c01 * (1 - weights[:, 2:3]) + c11 * weights[:, 2:3]

    # step 3
    c = c0 * (1 - weights[:, 3:4]) + c1 * weights[:, 3:4]

    return c


BOX_OFFSETS = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                           device='cuda')


class HashEmbedder(nn.Module):
    def __init__(self,
                 n_indim=3,
                 n_levels=16,
                 n_features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = np.exp((np.log(self.finest_resolution) - np.log(self.base_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList([nn.Embedding(2 ** self.log2_hashmap_size,
                                                      self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

        self.n_indim = n_indim
        if n_indim == 2:
            self.register_buffer('box_offset',
                                 torch.tensor([[i, j] for i in [0, 1] for j in [0, 1]]))
        elif n_indim == 3:
            self.register_buffer('box_offset',
                                 torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]))
        elif n_indim == 4:
            self.register_buffer('box_offset',
                                 torch.tensor([[[[i, j, k, l] for i in [0, 1]
                                                 for j in [0, 1] for k in [0, 1] for l in [0, 1]]]]))
        else:
            raise NotImplementedError(f"dim = {n_indim} not implemented")

    def get_voxel_vertices(self, xyz, resolution, log2_hashmap_size):
        '''
        xyz: 3D coordinates of samples. B x 3, should be inside [-1, 1]
        bounding_box: min and max x,y,z coordinates of object bbox
        resolution: number of voxels per axis
        '''
        grid_size = 2 / resolution
        xyz = xyz.clamp(-1, 1)
        bottom_left_idx = torch.floor((xyz + 1) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size - 1
        voxel_max_vertex = voxel_min_vertex + grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + self.box_offset
        hashed_voxel_indices = hash_coord(voxel_indices, log2_hashmap_size)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        if self.n_indim == 2:
            interp = bilinear_interp
        elif self.n_indim == 3:
            interp = trilinear_interp
        elif self.n_indim == 4:
            interp = quadlinear_interp
        else:
            interp = None

        for i in range(self.n_levels):
            resolution = np.floor(self.base_resolution * self.b ** i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = self.get_voxel_vertices(
                x, resolution, self.log2_hashmap_size)

            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)


class HashEmbedderFaster(nn.Module):
    def __init__(self,
                 n_indim=3,
                 n_levels=16,
                 n_features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 finest_resolution=512):
        """
        the domain is defined in [-1, 1]
        """
        super(HashEmbedderFaster, self).__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = np.exp((np.log(self.finest_resolution) - np.log(self.base_resolution)) / (n_levels - 1))

        self.register_parameter("embeddings",
                                nn.Parameter(
                                    torch.empty((n_levels, 2 ** self.log2_hashmap_size, self.n_features_per_level),
                                                dtype=torch.float32),
                                    requires_grad=True
                                ))

        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings.data, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

        self.n_indim = n_indim
        if n_indim == 2:
            self.register_buffer('box_offset',
                                 torch.tensor([[i, j] for i in [0, 1] for j in [0, 1]]))
        elif n_indim == 3:
            self.register_buffer('box_offset',
                                 torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]))
        elif n_indim == 4:
            self.register_buffer('box_offset',
                                 torch.tensor([[[[i, j, k, l] for i in [0, 1]
                                                 for j in [0, 1] for k in [0, 1] for l in [0, 1]]]]))
        else:
            raise NotImplementedError(f"dim = {n_indim} not implemented")

    def get_voxel_vertices(self, xyz, resolution, log2_hashmap_size):
        '''
        xyz: 3D coordinates of samples. B x 3, should be inside [-1, 1]
        bounding_box: min and max x,y,z coordinates of object bbox
        resolution: number of voxels per axis
        '''
        grid_size = 2 / resolution
        xyz = xyz.clamp(-1, 1)
        bottom_left_idx = torch.floor((xyz + 1) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size - 1
        voxel_max_vertex = voxel_min_vertex + grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + self.box_offset
        hashed_voxel_indices = hash_coord(voxel_indices, log2_hashmap_size)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def get_voxel(self, xyz, resolutions, log2_hashmap_size):
        '''
        xyz: 3D coordinates of samples. B x 3, should be inside [-1, 1]
        resolution: number of voxels per axis
        '''
        grid_size = (2 / resolutions.float())[:, None, None]  # r, 1, 1
        xyz = xyz.clamp(-1, 1)

        bottom_left_idx = torch.floor((xyz + 1)[None, ...] / grid_size).int()  # r, N, 3
        voxel_min_vertex = bottom_left_idx * grid_size - 1
        voxel_max_vertex = voxel_min_vertex + grid_size

        voxel_indices = bottom_left_idx.unsqueeze(-2) + self.box_offset[None, ...]  # r, N, 8, 3
        hashed_voxel_indices = hash_coord(voxel_indices, log2_hashmap_size)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        if self.n_indim == 2:
            interp = bilinear_interp
        elif self.n_indim == 3:
            interp = trilinear_interp
        elif self.n_indim == 4:
            interp = quadlinear_interp
        else:
            interp = None

        # for i in range(self.n_levels):
        #     resolution = np.floor(self.base_resolution * self.b ** i)
        #     voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = self.get_voxel_vertices(
        #         x, resolution, self.log2_hashmap_size)
        #
        #     voxel_embedds = self.embeddings[i][hashed_voxel_indices]
        #
        #     x_embedded = interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
        #     x_embedded_all.append(x_embedded)
        # voxel_embeded0 = torch.cat(x_embedded_all, dim=-1)

        # batchify executionX
        x = x.clamp(-1, 1)
        batch_sz = len(x)
        i = torch.arange(self.n_levels)
        resolutions = torch.floor(self.base_resolution * self.b ** i).to(x.device)
        voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices = self.get_voxel(
            x, resolutions, self.log2_hashmap_size)

        voxel_embedds = self.embeddings[i[:, None, None], hashed_voxel_indices]
        voxel_embeded = interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
        voxel_embeded = voxel_embeded.reshape(self.n_levels, batch_sz, -1).permute(1, 0, 2).reshape(batch_sz, -1)

        return voxel_embeded


if __name__ == "__main__":
    # test
    embedder = HashEmbedderFaster(3)
    x = torch.linspace(-1, 1, 102).reshape(-1, 3)
    embed = embedder(x)
    print(embed)
