import argparse

def get_args(cmd=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='RollLong-v1')
    parser.add_argument('--exp_prefix', type=str, default='1026_Roll_BC_Image')
    parser.add_argument('--num_env', type=int, default=1)  # Number of parallel environment
    parser.add_argument('--algo', type=str, default='imitation')
    parser.add_argument('--dataset_name', type=str, default='tmp')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--frame", type=str, default='world')
    parser.add_argument("--gd_num_steps", type=int, default=50, help="steps for the gradient descent(gd) expert")

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.02)  # For the solver
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam', choices=['Adam', 'Momentum'])
    parser.add_argument("--num_trajs", type=int, default=20)  # Number of demonstration trajectories
    parser.add_argument("--energy_weight", type=float, default=0.)
    parser.add_argument("--vel_loss_weight", type=float, default=0.)
    parser.add_argument("--opt_mode", type=str, default='naive')
    parser.add_argument("--num_resets", type=int, default=0)

    # Train
    parser.add_argument("--use_pcl", type=str, default='full_pcl')
    parser.add_argument("--il_num_epoch", type=int, default=5000)
    parser.add_argument("--il_lr", type=float, default=1e-3)
    parser.add_argument("--il_eval_freq", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--step_per_epoch", type=int, default=500)
    parser.add_argument("--step_warmup", type=int, default=2000)
    parser.add_argument("--hindsight_goal_ratio", type=float, default=0.5)
    parser.add_argument("--debug_overfit_test", type=bool, default=False)
    parser.add_argument("--obs_noise", type=float, default=0.05)
    parser.add_argument("--resume_path", default=None)
    parser.add_argument("--num_tools", type=int, default=1)
    parser.add_argument("--traj_limit", type=int, default=-1)
    parser.add_argument("--eval_train", type=bool, default=True)
    parser.add_argument("--rm_eval_noise", type=bool, default=True)
    parser.add_argument("--actor_type", type=str, default='Point')
    parser.add_argument("--buffer_horizon", type=int, default=50)
    parser.add_argument("--gt_tool", type=bool, default=False)
    parser.add_argument("--open_loop", type=bool, default=False)

    # Architecture
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--img_mode", type=str, default='rgbd')
    parser.add_argument("--pos_ratio", type=float, default=0.5)
    parser.add_argument("--pos_reset_ratio", type=float, default=0.2)  # 20% of the positive goals will come from the reset motion
    parser.add_argument("--z_dim", type=int, default=32)  # Maybe try multiple values
    parser.add_argument("--actor_feature_dim", type=int, default=128)
    parser.add_argument("--encoder_beta", type=float, default=10.)
    parser.add_argument("--bin_succ", type=bool, default=False)

    # Plan
    parser.add_argument("--adam_sample", type=int, default=400)
    parser.add_argument("--adam_iter", type=int, default=3000)
    parser.add_argument("--adam_lr", type=float, default=5e-2)
    parser.add_argument("--min_zlogl", type=float, default=-30)
    parser.add_argument("--save_goal_his", type=bool, default=False)
    parser.add_argument("--plan_step", type=int, default=2)

    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    return args