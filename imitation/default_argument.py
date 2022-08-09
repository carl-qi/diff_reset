import argparse


def get_args(cmd=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='PushSpread-v1')
    parser.add_argument('--algo', type=str, default='imitation')
    parser.add_argument('--dataset_name', type=str, default='tmp')
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_env", type=int, default=1)
    # Solver parameters
    parser.add_argument("--gd_max_iter", type=int, default=50, help="steps for the gradient descent(gd) expert")
    parser.add_argument("--lr", type=float, default=0.02)


    # Simulator parameters

    # Data generation parameters
    parser.add_argument("--num_trajs", type=int, default=20)

    # Actor
    parser.add_argument("--feature_dim", type=int, default=50)
    parser.add_argument("--il_num_epoch", type=int, default=1000)
    parser.add_argument("--il_lr", type=float, default=1e-3)
    parser.add_argument("--il_eval_freq", type=int, default=10)
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--img_mode", type=str, default='rgbd')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--frame_stack", type=int, default=1)

    # feasibility
    parser.add_argument("--pos_ratio", type=float, default=0.5)
    parser.add_argument("--bin_succ", type=bool, default=False)

    # encoder (VAE)
    parser.add_argument("--encoder_lr", type=float, default=1e-3)

    parser.add_argument("--debug_overfit_test", type=bool, default=False)
    if cmd:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    return args
