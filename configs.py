import configargparse


def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--datadir", type=str,
                        help='input data directory')
    parser.add_argument("--expdir", type=str,
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--seed", type=int, default=666,
                        help='random seed')

    # for mpi
    parser.add_argument("--mpi_h_scale", type=float, default=1.4,
                        help='the height of the stored MPI is <mpi_h_scale * H>')
    parser.add_argument("--mpi_w_scale", type=float, default=1.4,
                        help='the width of the stored MPI is <mpi_w_scale * W>')
    parser.add_argument("--mpi_h_verts", type=int, default=12,
                        help='the height of the stored MPI is <mpi_h_scale * H>')
    parser.add_argument("--mpi_w_verts", type=int, default=15,
                        help='the width of the stored MPI is <mpi_w_scale * W>')
    parser.add_argument("--mpi_d", type=int, default=64,
                        help='number of the MPI layer')
    parser.add_argument("--atlas_grid_h", type=int, default=8,
                        help='atlas_grid_h * atlas_grid_w == mpi_d')
    parser.add_argument("--atlas_size_scale", type=float, default=1,
                        help='atlas_size = mpi_d * H * W * atlas_size_scale')
    parser.add_argument("--model_type", type=str, default="MPI",
                        choices=["MPI", "MPMesh"])
    parser.add_argument("--optimize_depth", action='store_true',
                        help='if true, optimzing the depth of each plane')
    parser.add_argument("--optimize_normal", action='store_true',
                        help='if true, optimzing the normal of each plane')
    parser.add_argument("--optimize_geo_start", type=int, default=100000,
                        help='iteration to start optimizing verts and uvs')
    parser.add_argument("--optimize_verts_gain", type=float, default=1,
                        help='set 0 to disable the vertices optimization')
    parser.add_argument("--optimize_uvs_gain", type=float, default=1,
                        help='set 0 to disable the uvs optimization')

    parser.add_argument("--rgb_smooth_loss_weight", type=float, default=0,
                        help='rgb smooth loss')
    parser.add_argument("--a_smooth_loss_weight", type=float, default=0,
                        help='rgb smooth loss')
    parser.add_argument("--arap_loss_weight", type=float, default=0,
                        help='as rigid as possible smooth loss')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--batch_size", type=int, default=1024*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=30,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=50000)
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,
                        help='number of additional fine samples per ray')
    parser.add_argument("--N_samples_fine", type=int, default=64,
                        help='n sample fine = N_samples_fine + N_importance')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", type=bool, default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--embed_type", type=str, default='pe',
                        help='pe, none, hash, dict')
    parser.add_argument("--log2_embed_hash_size", type=int, default=19,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_window_start", type=int, default=0,
                        help='windowed PE start step')
    parser.add_argument("--multires_window_end", type=int, default=-1,
                        help='windowed PE end step, negative to disable')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_views_window_start", type=int, default=0,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_views_window_end", type=int, default=-1,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1e0,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_rgba", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_texture", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_view", type=int, default=-1,
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_slice", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_keypoints", action='store_true',
                        help='change the keypoint location')
    parser.add_argument("--render_deformed", type=str, default='',
                        help='edited file')
    parser.add_argument("--render_factor", type=float, default=1,
                        help='change the keypoint location')
    parser.add_argument("--render_canonical", action='store_true',
                        help='if true, the DNeRF is like traditional NeRF')

    ## data options
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # logging options
    parser.add_argument("--i_img",    type=int, default=300,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_print",   type=int, default=300,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=20000,
                        help='frequency of testset saving')
    parser.add_argument("--i_eval", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')

    # multiprocess learning
    parser.add_argument("--gpu_num", type=int, default='-1', help='number of processes')

    # test use latent_t
    parser.add_argument("--render_chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # MA Li in General
    # ===================================
    parser.add_argument("--frm_num", type=int, default=-1, help='number of frames to use')
    parser.add_argument("--bd_factor", type=float, default=0.65, help='expand factor of the ROI box')
    parser.add_argument("--optimize_poses", default=False, action='store_true',
                        help='optimize poses')
    parser.add_argument("--optimize_poses_start", type=int, default=0, help='start step of optimizing poses')
    parser.add_argument("--surpress_boundary_thickness", type=int, default=0,
                        help='do not supervise the boundary of thickness <>, 0 to disable')
    parser.add_argument("--itertions_per_frm", type=int, default=50)
    parser.add_argument("--masked_sample_precent", type=float, default=0.92,
                        help="in batch_size samples, precent of the samples that are sampled at"
                             "masked region, set to 1 to disable the samples on black region")
    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--sigma_activate", type=str, default='relu',
                        help='activate function for sigma output, choose among "relu", "softplus",'
                             '"volsdf"')
    parser.add_argument("--use_raw2outputs_old", type=bool, default=True,
                        help='use the original raw2output (not forcing the last layer to be alpha=1')
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='if true, nerf_coarse == nerf_fine')
    parser.add_argument("--not_supervise_rgb0", action='store_true', default=False,
                        help='if true, rgb0 well not considered as part of the loss')
    parser.add_argument("--best_frame_idx", type=int, default=-1,
                        help='if > 0, the first epoch will be trained only on this frame')

    ## For other losses
    parser.add_argument("--sparsity_type", type=str, default='none',
                        help='sparsity loss type, choose among none, l1, l1/l2, entropy')
    parser.add_argument("--sparsity_loss_weight", type=float, default=0,
                        help='sparsity loss weight')
    parser.add_argument("--sparsity_loss_start_step", type=float, default=50000,
                        help='sparsity loss weight')
    parser.add_argument("--cycle_loss_weight", type=float, default=0,
                        help='cycle loss of neutex')
    parser.add_argument("--cycle_loss_decay", type=float, default=0,
                        help='cycle loss of neutex')
    parser.add_argument("--cyclenetdepth", type=int, default=5,
                        help='layers in network')
    parser.add_argument("--cyclenetwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--cyclenet_time_layeridx", type=int, default=0,
                        help='the layer index when introducing time embedding')
    parser.add_argument("--use_two_time_for_cycle", type=bool, default=False,
                        help='pe or latent')
    parser.add_argument("--time_embed_type_for_cycle", type=str, default="latent",
                        help='pe or latent')
    parser.add_argument("--latent_size_for_cycle", type=int, default=0,
                        help='latent_size')
    parser.add_argument("--time_multires_for_cycle", type=int, default=5,
                        help='embedding dim on time axis')
    parser.add_argument("--use_two_embed_for_cycle", type=bool, default=False,
                        help='use two individual embedding for input xyz')
    parser.add_argument("--multires_for_cycle", type=int, default=0,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--log2_hash_size_for_cycle", type=int, default=14,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--embed_type_for_cycle", type=str, default='pe',
                        help='embed type for cycle')
    # parser.add_argument("--multires_window_start_for_cycle", type=int, default=0,
    #                     help='log2 of max freq for positional encoding (3D location)')
    # parser.add_argument("--multires_window_end_for_cycle", type=int, default=-1,
    #                     help='log2 of max freq for positional encoding (3D location)')

    parser.add_argument("--alpha_loss_weight", type=float, default=0,
                        help='alpha sparsity loss')
    parser.add_argument("--alpha_loss_decay", type=int, default=1e10,
                        help='alpha sparsity loss decay')
    parser.add_argument("--alpha_type", type=str, default="add",
                        help='choose among add & multiply')
    parser.add_argument("--smooth_loss_weight", type=float, default=0,
                        help='edge-aware disparity smooth loss')
    parser.add_argument("--smooth_loss_start_decay", type=int, default=1e10,
                            help='edge-aware disparity smooth loss start step')
    parser.add_argument("--temporal_loss_weight", type=float, default=0,
                        help='temporal consistency loss')
    parser.add_argument("--temporal_loss_start_step", type=float, default=0,
                        help='temporal consistency loss start step')
    parser.add_argument("--temporal_loss_patch_num", type=int, default=100,
                        help='patch number of temporal loss, the actual number will multiply by the gpu_num')

    ## For UVField
    parser.add_argument("--uv_activate", type=str, default='tanh',
                        help='activate function for uv output, choose among "tanh"')
    parser.add_argument("--uv_map_gt_skip_num", type=int, default=0,
                        help='will only load pos map every <skip_num> frames')
    parser.add_argument("--uv_batch_size", type=int, default=1024,
                        help='batch size of the uv supervision')
    parser.add_argument("--uv_loss_weight", type=float, default=0,
                        help='batch size of the uv supervision, controling when to enter uv_loss branch')
    parser.add_argument("--uv_loss_decay", type=int, default=10,
                        help='batch size of the uv supervision')
    parser.add_argument("--uv_loss_noise_std", type=float, default=0.002,
                        help='add randn noise to the supervision location of the uv_loss')
    parser.add_argument("--use_two_texmodels_for_fine", action='store_true',
                        help='if true, nerf_coarse == nerf_fine')
    parser.add_argument("--uv_map_face_roi", type=float, default=0.5,
                        help='percent of the face texture in the all texture')

    parser.add_argument("--uvsmooth_loss_weight", type=float, default=0,
                        help='smoothness for the uv field')
    parser.add_argument("--uvprepsmooth_loss_weight", type=float, default=0,
                        help='smoothness for the uv field project to surface')
    parser.add_argument("--dsmooth_loss_weight", type=float, default=0,
                        help='smoothness for the density raw field')

    parser.add_argument("--texture_type", type=str, default="map",
                        help='type of texure, choose among map, mlp, fuse')
    parser.add_argument("--promote_fuse_texture_step", type=int, default=1e15,
                        help='convert the mlp to texture map')
    parser.add_argument("--freeze_uvfield_step", type=int, default=1e15,
                        help='freeze the uv field')

    # For texture_type == map
    parser.add_argument("--texture_map_channel", type=int, default=3,
                        help='number of channels in the texture map, negative to disable')
    parser.add_argument("--texture_map_resolution", type=int, default=1024,
                        help='number of channels in the texture map, negative to disable')
    parser.add_argument("--texture_map_gradient_multiply", type=int, default=1,
                        help='gradient of the texture map will be multiply by this value')
    parser.add_argument("--texture_map_ini", type=str, default='',
                        help='initialization of the texture map')
    parser.add_argument("--texture_map_post", type=str, default='',
                        help='only use when render_only is true')
    parser.add_argument("--texture_map_post_isfull", action='store_true', default=True,
                        help='load texture map as full texture')
    parser.add_argument("--texture_map_force_map", action='store_true',
                        help='load texture map as full texture')
    parser.add_argument("--geometry_map_post", type=str, default='',
                        help='only use when render_only is true')
    parser.add_argument("--geometry_map_post_isfull", action='store_true',
                        help='load texture map as full texture')

    # For texture_type == mlp
    parser.add_argument("--texnetdepth", type=int, default=6,
                        help='layers in network')
    parser.add_argument("--texnetwidth", type=int, default=512,
                        help='channels per layer')
    parser.add_argument("--texnet_view_layeridx", type=int, default=-2,
                        help='the layer index when introducing view embedding')
    parser.add_argument("--texnet_time_layeridx", type=int, default=0,
                        help='the layer index when introducing time embedding')
    parser.add_argument("--tex_multires", type=int, default=10,
                        help='channels per layer')
    parser.add_argument("--tex_embed_type", type=str, default='pe',
                        help='type of tex embed')
    parser.add_argument("--tex_log2_hash_size", type=int, default=14,
                        help='type of tex embed')

    parser.add_argument("--use_two_time_for_tex", type=bool, default=False,
                        help='pe or latent')
    parser.add_argument("--time_embed_type_for_tex", type=str, default="latent",
                        help='pe or latent')
    parser.add_argument("--latent_size_for_tex", type=int, default=0,
                        help='latent_size')
    parser.add_argument("--time_multires_for_tex", type=int, default=5,
                        help='embedding dim on time axis')
    parser.add_argument("--time_multires_window_start_for_tex", type=int, default=0,
                        help='embedding dim on time axis')
    parser.add_argument("--time_multires_window_end_for_tex", type=int, default=-1,
                        help='embedding dim on time axis')

    # For NeRFModulate
    parser.add_argument("--time_embed_type", type=str, default="latent",
                        help='pe or latent')
    parser.add_argument("--latent_size", type=int, default=0,
                        help='latent_size')

    # For NeRFTemporal
    parser.add_argument("--time_multires", type=int, default=5,
                        help='embedding dim on time axis')
    parser.add_argument("--use_two_dmodels_for_fine", action='store_true',
                        help='if true, nerf_coarse == nerf_fine')
    parser.add_argument("--dnetdepth", type=int, default=7,
                        help='layers in network')
    parser.add_argument("--dnetwidth", type=int, default=128,
                        help='channels per layer')
    parser.add_argument("--ambient_slicing_dim", type=int, default=0,
                        help='channels per layer')
    parser.add_argument("--slice_multires", type=int, default=7,
                        help='channels per layer')
    parser.add_argument("--slicenetdepth", type=int, default=6,
                        help='layers in network')
    parser.add_argument("--slicenetwidth", type=int, default=128,
                        help='channels per layer')

    # Towards geometry editing
    parser.add_argument("--density_type", type=str, default='direct',
                        help='choose among direct, xyz_norm')
    parser.add_argument("--explicit_warp_type", type=str, default='none',
                        help='choose among none, rigid, prnet')
    parser.add_argument("--canonicaldir", type=str, default='/apdcephfs/private_leema/data/NeRFtx/PRNet/Data/uv-data/canonical_vertices_my.npy',
                        help='path to canonical dir')
    parser.add_argument("--kptidsdir", type=str, default='/apdcephfs/private_leema/data/NeRFtx/PRNet/Data/uv-data/kpts2.npy',
                        help='path to canonical dir')
    parser.add_argument("--uvweightdir", type=str, default='/apdcephfs/private_leema/data/NeRFtx/PRNet/Data/uv-data/face_uv_mask.png',
                        help='path to canonical dir')
    parser.add_argument("--model_affine", action='store_true', default=False,
                        help='whether to fit per control point affine transform')
    parser.add_argument("--rbf_perframe", action='store_true', default=False,
                        help='whether to fit per frame rbf')
    parser.add_argument("--kpt_loss_weight", type=float, default=0,
                        help='make the explicit warp more temporal smooth')
    parser.add_argument("--gsmooth_loss_weight", type=float, default=0,
                        help='make the explicit warp more temporal smooth')
    parser.add_argument("--gsmooth_loss_type", type=str, default='o1',
                        help='o1 or o2, order n derivative')
    parser.add_argument("--gsmooth_loss_decay", type=int, default=10000,
                        help='make the explicit warp more temporal smooth')

    # not being considered anymore
    parser.add_argument("--test_ids", type=str, default='4',
                        help='example: 3,4,5,6  splited by \',\'')
    parser.add_argument("--uv_map_gt", type=str, default='',
                        help='deprcated')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    return parser
