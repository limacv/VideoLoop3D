import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # Two sets of config for naive hierarchical config structure
    parser.add_argument('--config', is_config_file=True,
                        help='config file path for base')
    parser.add_argument('--config1', is_config_file=True, default='',
                        help='config file path for each data')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--expname_postfix", type=str, default='',
                        help='experiment name = expname + expname_postfix')
    parser.add_argument("--test_view_idx", type=str, default='',
                        help='#,#,#')

    parser.add_argument("--prefix", type=str, default='',
                        help='the root of everything')
    parser.add_argument("--datadir", type=str,
                        help='input data directory')
    parser.add_argument("--expdir", type=str,
                        help='where to store ckpts and logs')
    parser.add_argument("--seed", type=int, default=666,
                        help='random seed')
    parser.add_argument("--factor", type=int, default=2,
                        help='downsample factor for LLFF images')
    parser.add_argument("--near_factor", type=float, default=0.9, help='the actual near plane will be near_factor * near')
    parser.add_argument("--far_factor", type=float, default=2, help='the actual far plane will be far_factor * far')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='unused')
    parser.add_argument("--fp16", action='store_true',
                        help='use half precision to train, currently still have bug, do NOT use')
    parser.add_argument("--bg_color", type=str, default="",
                        help='0#0#0, or random, the background color')
    parser.add_argument("--scale_invariant", action='store_true',
                        help='scale_invariant rgb loss, scaling before compute the MSE')

    # for MPV only, not used for MPMesh
    parser.add_argument("--mpv_frm_num", type=int, default=90,
                        help='frame number of the mpv')
    parser.add_argument("--mpv_isloop", action='store_true',
                        help='whether to produce looping videos')
    parser.add_argument("--init_from", type=str, default='',
                        help='path to ckpt, will add prefix, currently only support reload from MPI')
    parser.add_argument("--init_std", type=float, default=0,
                        help='noise std of the dynamic MPV')
    parser.add_argument("--add_uv_noise", action='store_true',
                        help='add noise to uv, unused')
    parser.add_argument("--add_intrin_noise", action='store_true',
                        help='add noise to intrinsic, to prevent tiling artifact')

    # loss config
    parser.add_argument("--loss_ref_idx", type=str, default='0',
                        help='#,#,# swd_alpha = ref if view==swd_alpha_reference_viewidx else other')
    parser.add_argument("--loss_name", type=str, default='gpnn',
                        help='gpnn, mse, swd, avg. gpnn_x to specify alpha==x')
    parser.add_argument("--loss_name_ref", type=str, default='gpnn',
                        help='gpnn, mse, swd, avg. gpnn_x to specify alpha==x')
    parser.add_argument("--swd_macro_block", type=int, default=65,
                        help='used for gpnn low mem')
    parser.add_argument("--swd_patch_size_ref", type=int, default=5,
                        help='gpnn patch size for reference view')
    parser.add_argument("--swd_patch_size", type=int, default=5,
                        help='gpnn patch size for other view')
    parser.add_argument("--swd_patcht_size_ref", type=int, default=5,
                        help='gpnn temporal patch size for reference view')
    parser.add_argument("--swd_patcht_size", type=int, default=5,
                        help='gpnn temporal patch size for other view')
    parser.add_argument("--swd_stride_ref", type=int, default=2,
                        help='gpnn stride size for reference view')
    parser.add_argument("--swd_stride", type=int, default=2,
                        help='gpnn stride size for other view')
    parser.add_argument("--swd_stridet", type=int, default=2,
                        help='gpnn temporal stride size for reference view')
    parser.add_argument("--swd_stridet_ref", type=int, default=2,
                        help='gpnn temporal stride size for other view')
    parser.add_argument("--swd_rou", type=str, default='0',
                        help='parameter of robustness term, can also be mse, abs')
    parser.add_argument("--swd_rou_ref", type=str, default='0',
                        help='parameter of robustness term, can also be mse, abs')
    parser.add_argument("--swd_scaling", type=float, default=0.2,
                        help='parameter of robustness term')
    parser.add_argument("--swd_scaling_ref", type=float, default=0.2,
                        help='parameter of robustness term')
    parser.add_argument("--swd_alpha", type=float, default=0,
                        help='alpha, bigger than 100 is equivalent to None, (the rou in paper)')
    parser.add_argument("--swd_alpha_ref", type=float, default=0,
                        help='alpha, bigger than 100 is equivalent to None, (the rou in paper)')
    parser.add_argument("--swd_dist_fn", type=str, default='mse',
                        help='distance function, currently not setable')
    parser.add_argument("--swd_dist_fn_ref", type=str, default='mse',
                        help='distance function, currently not setable')
    parser.add_argument("--swd_factor", type=int, default=1,
                        help='factor, will compute NN in factored images')
    parser.add_argument("--swd_factor_ref", type=int, default=1,
                        help='factor, will compute NN in factored images')
    parser.add_argument("--swd_loss_gain_ref", type=float, default=1,
                        help='alpha, bigger than 100 is equivalent to None')

    # pyramid configuration
    parser.add_argument("--pyr_stage", type=str, default='',
                        help='x,y,z,...   iteration to upsample')
    parser.add_argument("--pyr_minimal_dim", type=int, default=60,
                        help='if > 0, will determine the pyr_stage')
    parser.add_argument("--pyr_num_epoch", type=int, default=600,
                        help='iter num in each level')
    parser.add_argument("--pyr_factor", type=float, default=0.5,
                        help='factor in each pyr level')
    parser.add_argument("--pyr_init_level", type=int, default=-1,
                        help='before that, use mse')

    # for mpi
    parser.add_argument("--sparsify_epoch", type=int, default=-1,
                        help='sparsify the MPMesh in epoch')
    parser.add_argument("--sparsify_rmfirstlayer", type=int, default=0,
                        help='if true, will remove the first #i layer')
    parser.add_argument("--sparsify_erode", type=int, default=2,
                        help='iters to dilate the alpha channel')
    parser.add_argument("--learn_loop_mask", action='store_true',
                        help='if true, will learn a loop_mask jointly')

    parser.add_argument("--direct2sh_epoch", type=int, default=-1,
                        help='converting direct to sh, unused now')
    parser.add_argument("--sparsify_alpha_thresh", type=float, default=0.03,
                        help='alpha thresh for tile culling')
    parser.add_argument("--vid2img_mode", type=str, default='average',
                        help='choose among average, median, static, dynamic')
    parser.add_argument("--mpi_h_scale", type=float, default=1,
                        help='the height of the stored MPI is <mpi_h_scale * H>')
    parser.add_argument("--mpi_w_scale", type=float, default=1,
                        help='the width of the stored MPI is <mpi_w_scale * W>')
    parser.add_argument("--mpi_h_verts", type=int, default=12,
                        help='number of vertices, decide the tile size')
    parser.add_argument("--mpi_w_verts", type=int, default=15,
                        help='number of vertices, decide the tile size')
    parser.add_argument("--mpi_d", type=int, default=64,
                        help='number of the MPI layer')
    parser.add_argument("--atlas_grid_h", type=int, default=8,
                        help='atlas_grid_h * atlas_grid_w == mpi_d')
    parser.add_argument("--atlas_size_scale", type=float, default=1,
                        help='atlas_size = mpi_d * H * W * atlas_size_scale')
    parser.add_argument("--atlas_cnl", type=int, default=4,
                        help='channel num, currently not setable, much be 4')
    parser.add_argument("--model_type", type=str, default="MPMesh",
                        help='currently not setable, much be MPMesh')
    parser.add_argument("--rgb_mlp_type", type=str, default='direct',
                        help='not used, must be direct')
    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--alpha_activate", type=str, default='sigmoid',
                        help='activate function for alpha output, choose among "none", "sigmoid"')
    parser.add_argument("--optimize_geo_start", type=int, default=10000000,
                        help='iteration to start optimizing verts and uvs, currently not used')
    parser.add_argument("--optimize_verts_gain", type=float, default=1,
                        help='set 0 to disable the vertices optimization')
    parser.add_argument("--normalize_verts", action='store_true',
                        help='if true, the parameter is normalized')

    # about training
    parser.add_argument("--upsample_stage", type=str, default="",
                        help='x,y,z,...  stage to perform upsampling')
    parser.add_argument("--rgb_smooth_loss_weight", type=float, default=0,
                        help='rgb spatial smooth loss')
    parser.add_argument("--a_smooth_loss_weight", type=float, default=0,
                        help='alpha spatial smooth loss')
    parser.add_argument("--d_smooth_loss_weight", type=float, default=0,
                        help='depth smooth loss')
    parser.add_argument("--l_smooth_loss_weight", type=float, default=0,
                        help='loop mask (label) smooth loss')
    parser.add_argument("--edge_scale", type=float, default=4,
                        help='edge aware smooth loss, 0 to disable edge aware')
    parser.add_argument("--normalize_blendweight_fordepth", action='store_true',
                        help='edge aware smooth loss, 0 to disable edge aware')
    parser.add_argument("--density_loss_weight", type=float, default=0,
                        help='density loss')
    parser.add_argument("--density_loss_epoch", type=int, default=0,
                        help='gradually grow the density to epoch')
    parser.add_argument("--sparsity_loss_weight", type=float, default=0,
                        help='sparsity loss weight')

    # training options
    parser.add_argument("--N_iters", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'],
                        help='optmizer')
    parser.add_argument("--patch_h_size", type=int, default=512,
                        help='patch size for each iteration')
    parser.add_argument("--patch_w_size", type=int, default=512,
                        help='patch size for each iteration')
    parser.add_argument("--patch_h_stride", type=int, default=128,
                        help='stride size for each iteration')
    parser.add_argument("--patch_w_stride", type=int, default=128,
                        help='stride size for each iteration')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_adaptive", action='store_true',
                        help='adaptively adjust learning rate based on patch size, or it will generate noise')
    parser.add_argument("--lrate_decay", type=int, default=30,
                        help='exponential learning rate decay (in 1000 steps)')

    # logging options
    parser.add_argument("--i_img",    type=int, default=300,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_print",   type=int, default=300,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_video",   type=int, default=10000,
                        help='frequency of render_poses video saving')

    # multiprocess learning
    parser.add_argument("--gpu_num", type=int, default='-1', 
                        help='number of processes, currently only support 1 gpu')
    return parser
