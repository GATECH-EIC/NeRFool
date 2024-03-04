import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--rootdir",
        type=str,
        default="./",
        help="the path to the project root directory. Replace this path with yours!",
    )
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--distributed", action="store_true", help="if use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="rank for distributed training")
    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )

    ########## dataset options ##########
    ## train and eval dataset
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="ibrnet_collected",
        help="the training dataset, should either be a single dataset, "
        'or multiple datasets connected with "+", for example, ibrnet_collected+llff+spaces',
    )
    parser.add_argument(
        "--dataset_weights",
        nargs="+",
        type=float,
        default=[],
        help="the weights for training datasets, valid when multiple datasets are used.",
    )
    parser.add_argument(
        "--train_scenes",
        nargs="+",
        default=[],
        help="optional, specify a subset of training scenes from training dataset",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="llff_test", help="the dataset to evaluate"
    )
    parser.add_argument(
        "--eval_scenes",
        nargs="+",
        default=[],
        help="optional, specify a subset of scenes from eval_dataset to evaluate",
    )
    ## others
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, "
        "useful for large datasets like deepvoxels or nerf_synthetic",
    )

    ########## model options ##########
    ## ray sampling options
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="uniform",
        help="how to sample pixels from images for training:" "uniform|center",
    )
    parser.add_argument(
        "--center_ratio", type=float, default=0.8, help="the ratio of center crop to keep"
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 16,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 4,
        help="number of rays processed in parallel, decrease if running out of memory",
    )

    ## model options
    parser.add_argument(
        "--coarse_feat_dim", type=int, default=32, help="2D feature dimension for coarse level"
    )
    parser.add_argument(
        "--fine_feat_dim", type=int, default=32, help="2D feature dimension for fine level"
    )
    parser.add_argument(
        "--num_source_views",
        type=int,
        default=10,
        help="the number of input source views for each target view",
    )
    parser.add_argument(
        "--rectify_inplane_rotation", action="store_true", help="if rectify inplane rotation"
    )
    parser.add_argument("--coarse_only", action="store_true", help="use coarse network only")
    parser.add_argument(
        "--anti_alias_pooling", type=int, default=1, help="if use anti-alias pooling"
    )
    parser.add_argument("--trans_depth", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--netwidth", type=int, default=64, help="network intermediate dimension")
    parser.add_argument(
        "--single_net",
        type=bool,
        default=True,
        help="use single network for both coarse and/or fine sampling",
    )

    ########## checkpoints ##########
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--no_load_opt", action="store_true", help="do not load optimizer when reloading"
    )
    parser.add_argument(
        "--no_load_scheduler", action="store_true", help="do not load scheduler when reloading"
    )

    ########### iterations & learning rate options ##########
    parser.add_argument("--n_iters", type=int, default=250000, help="num of iterations")
    parser.add_argument(
        "--lrate_feature", type=float, default=1e-3, help="learning rate for feature extractor"
    )
    parser.add_argument("--lrate_gnt", type=float, default=5e-4, help="learning rate for gnt")
    parser.add_argument(
        "--lrate_decay_factor",
        type=float,
        default=0.5,
        help="decay learning rate by a factor every specified number of steps",
    )
    parser.add_argument(
        "--lrate_decay_steps",
        type=int,
        default=50000,
        help="decay learning rate by a factor every specified number of steps",
    )

    ########## rendering options ##########
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance", type=int, default=64, help="number of important samples per ray"
    )
    parser.add_argument(
        "--inv_uniform", action="store_true", help="if True, will uniformly sample inverse depths"
    )
    parser.add_argument(
        "--det", action="store_true", help="deterministic sampling for coarse and fine samples"
    )
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="apply the trick to avoid fitting to white background",
    )
    parser.add_argument(
        "--render_stride",
        type=int,
        default=1,
        help="render with large stride for validation to save time",
    )

    ########## logging/saving options ##########
    parser.add_argument("--i_print", type=int, default=100, help="frequency of terminal printout")
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )

    ########## evaluation options ##########
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--depth_var_loss", type=float, default=0,
                        help='regularize the depth variance per ray')

    parser.add_argument("--adv_iters", type=int, default=100,
                        help='number of iterations for updating the adv perturbations')

    parser.add_argument("--epsilon", type=int, default=8,
                        help='adversarial perturbation strength')

    parser.add_argument("--adv_lr", type=float, default=2,
                        help='learning rate for updating the perturbation')

    parser.add_argument('--use_clean_color', action='store_true', help='Use colors predicted on clean source views')

    parser.add_argument('--use_clean_density', action='store_true', help='Use density predicted on clean source views')

    parser.add_argument("--orig_dist_thres", type=float, default=-1,
                        help='the distance threshold for using the training view in the image-specific mode')

    parser.add_argument('--export_adv_source_img', action='store_true', help='Export adversarially perturbed source view images')

    parser.add_argument('--depth_smooth_loss', type=float, default=0, help='Weights of depth smooth loss')

    parser.add_argument("--patch_size", type=int, default=8,
                        help='patch size for depth smooth loss')

    parser.add_argument('--depth_consistency_loss', type=float, default=0, help='warp the depth of source views to the target view')

    parser.add_argument('--ds_rgb', action='store_true', help='downsample rgb images to match the resolution of depth; otherwise upsample depth')

    parser.add_argument('--depth_diff_loss', type=float, default=0, help='differences with the GT depth')

    parser.add_argument('--use_patch_sampling', action='store_true', help='Use random patch sampling for rays')

    parser.add_argument('--gt_depth_path', type=str, default='', help='path to ground-truth depth')

    parser.add_argument('--use_pseudo_gt', action='store_true', help='Use clean rendered rgbs as ground truth')

    parser.add_argument('--view_specific', action='store_true', help='optimize view-specific adv perturbations; otherwise optimize generalizable ones across different views')

    parser.add_argument('--use_unseen_views', action='store_true', help='optimize generalizable adv perturb with unseen views')

    parser.add_argument('--no_attack', action='store_true', help='do not use adv attack')

    parser.add_argument('--use_adam', action='store_true', help='use adam optimizer')

    parser.add_argument('--adam_lr', type=float, default=0, help='learning rate for adam')

    parser.add_argument('--lr_step_size', type=int, default=100, help='step lr size')

    parser.add_argument('--lr_gamma', type=float, default=0.5, help='step lr decay')

    parser.add_argument('--use_pcgrad', action='store_true', help='enable pcgrad')

    parser.add_argument('--major_loss', type=str, default='', help='major loss direction in pcgrad')

    parser.add_argument('--use_dp', action='store_true', help='enable data parallel')

    parser.add_argument('--density_loss', type=float, default=0, help='diff loss on the predicted density')

    parser.add_argument('--interp_upbound', type=float, default=1., help='upper bound of the sampling for the camera pose interplation')

    parser.add_argument('--decouple_interp_range', action='store_true', help='decouple the interpolation range of rotation and translation')

    parser.add_argument('--interp_upbound_rot', type=float, default=1., help='upper bound of the sampling for the rotation matrix interplation')

    parser.add_argument('--interp_upbound_trans', type=float, default=1., help='upper bound of the sampling for the translation interplation')

    parser.add_argument('--sample_based_on_depth', action='store_true', help='sample view directions based on the camera depth for interpolation unseen views')

    parser.add_argument('--beta', type=float, default=0.5, help='alpha/beta for the interpolation param sampled from the beta distribution')

    parser.add_argument('--temp', type=float, default=0.5, help='temperature for camera pose sampling')

    parser.add_argument('--perturb_camera', action='store_true', help='adversarially perturb the camera poses')

    parser.add_argument('--perturb_camera_no_opt', action='store_true', help='randomly perturb the camera poses with optimization (an ablation study)')

    parser.add_argument('--perturb_camera_no_detach', action='store_true', help='jointly update camera and rets')

    parser.add_argument('--zero_camera_init', action='store_true', help='zero initialization for camera perturbations)')

    parser.add_argument('--rot_epsilon', type=float, default=10, help='epsilons for perturbing camera rotation degree (in terms of degree)')

    parser.add_argument('--trans_epsilon', type=float, default=0.1, help='epsilons for perturbing camera translation distances')

    parser.add_argument('--camera_consistency_loss', type=float, default=0, help='multiview consistency loss for perturbing the camera poses')

    parser.add_argument('--cam_src2tar', type=float, default=0, help='multiview consistency loss for perturbing the camera poses')

    parser.add_argument('--cam_tar2src', type=float, default=0, help='multiview consistency loss for perturbing the camera poses')

    parser.add_argument('--cam_depth', type=float, default=0, help='multiview consistency loss for perturbing the camera poses')

    parser.add_argument('--use_adv_train', action='store_true', help='enable adversarial training')

    parser.add_argument('--geo_noise', type=float, default=0, help='std of the gaussian noise applied on sigma')

    parser.add_argument('--use_purification', action='store_true', help='apply purifications on the adversarially perturbed source views')

    parser.add_argument("--purif_epsilon", type=int, default=8, help='purification strength')

    parser.add_argument("--purif_lr", type=float, default=2, help='learning rate for updating the purification')

    parser.add_argument("--purif_iters", type=int, default=100, help='number of iterations for updating the purification')

    parser.add_argument('--purif_consistency_loss', type=float, default=0, help='multiview consistency loss for optimizing the purification')

    parser.add_argument('--use_self_purification', action='store_true', help='reconstruct source views as a way of purification')

    parser.add_argument('--def_random_noise', type=float, default=0, help='use random noise in terms of pixels as a way of defense')

    parser.add_argument('--ret_alpha', action='store_true', help="return alpha with rgb")

    parser.add_argument('--use_trans_attack', action='store_true', help='transfer the attack optimized for one view to attack another view')

    parser.add_argument("--total_view_limit", type=int, default=None, help='limit the total numbers of training and test views')

    return parser
