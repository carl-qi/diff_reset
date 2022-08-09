def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    from imitation.train_full import get_args, set_random_seed
    from plb.envs.mp_wrapper import make_mp_envs

    args = get_args(cmd=False)
    args.resume_path = 'data/autobot/0929_GatherMove_train_hindsight/0929_GatherMove_train_hindsight/0929_GatherMove_train_hindsight_2021_09_29_01_17_30_0014/agent_60.ckpt'
    args.env_name = 'GatherMove-v1'

    set_random_seed(args.seed)

    device = 'cuda'
    env = make_mp_envs(args.env_name, args.num_env, args.seed)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from imitation.agent import Agent
    from imitation.eval_helper import eval_plan
    from chester import logger
    import json
    vv_path = os.path.join(os.path.dirname(args.resume_path), 'variant.json')
    with open(vv_path, 'r') as f:
        vv = json.load(f)
    args.__dict__.update(**vv)
    args.__dict__.update(**arg_vv)
    args.resume_path = 'data/autobot/0929_GatherMove_train_hindsight/0929_GatherMove_train_hindsight/0929_GatherMove_train_hindsight_2021_09_29_01_17_30_0014/agent_60.ckpt'

    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)

    print(args.cached_state_path)
    obs_channel = len(args.img_mode) * args.frame_stack
    img_obs_shape = (args.image_dim, args.image_dim, obs_channel)
    action_dim = env.getattr('taichi_env.primitives.action_dim')[0]

    agent = Agent(args, None, img_obs_shape, action_dim, num_tools=args.num_tools, device=device)
    agent.load(args.resume_path)

    logger.configure(dir='./data/iclr_tune/' + args.save_name, exp_name='GatherMove-1')
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)

    result = eval_plan(args, env, agent, 0, demo=True)
    logger.log(result)
