from imitation.agent import Agent
from imitation.imitation_buffer import ImitationReplayBuffer
from imitation.utils import aggregate_traj_info
from tqdm import tqdm
import json
import os
from chester import logger
#
from plb.envs.mp_wrapper import make_mp_envs
from imitation.utils import load_target_info
from imitation.eval_helper import eval_skills, eval_vae, eval_plan
from imitation.train_full import get_args, set_random_seed


def run_task(arg_vv, log_dir, exp_name):  # Chester launch
    args = get_args(cmd=False)

    args.__dict__.update(**arg_vv)
    args.num_tools = 1

    set_random_seed(args.seed)

    # # Configure logger
    logger.configure(dir=log_dir, exp_name=exp_name)
    log_dir = logger.get_dir()
    assert log_dir is not None
    os.makedirs(log_dir, exist_ok=True)
    #
    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # Need to make the environment before moving tensor to torch
    obs_channel = len(args.img_mode) * args.frame_stack
    img_obs_shape = (args.image_dim, args.image_dim, obs_channel)
    env = make_mp_envs(args.env_name, args.num_env, args.seed)

    args.cached_state_path = env.getattr('cfg.cached_state_path', 0)
    print(args.cached_state_path)
    action_dim = env.getattr('taichi_env.primitives.action_dim')[0]

    # Load buffer
    device = 'cuda'
    buffer = ImitationReplayBuffer(args)
    buffer.load(args.dataset_path)
    buffer.generate_train_eval_split()
    target_info = load_target_info(args, device)
    buffer.__dict__.update(**target_info)

    # # ----------preparation done------------------
    agent = Agent(args, None, img_obs_shape, action_dim, num_tools=args.num_tools, device=device)
    if args.resume_path is not None:
        agent.load(args.resume_path)

    for epoch in range(args.il_num_epoch):
        epoch_tool_idxes = [buffer.get_epoch_tool_idx(epoch, tid) for tid in [0, 1]]
        train_infos = []
        for batch_tools_idx in tqdm(zip(*epoch_tool_idxes)):
            data_batch = buffer.sample_tool_transitions(batch_tools_idx, epoch, device)
            train_info = agent.train(data_batch, agent_ids=[0, 0])
            train_infos.append(train_info)

        if epoch % args.il_eval_freq == 0:
            # Log training info
            train_infos = aggregate_traj_info(train_infos, prefix=None)

            # Evaluate skills
            skill_traj, skill_info = eval_skills(args, env, agent, epoch, tids=[0])
            vae_info = eval_vae(args, agent, skill_traj, epoch)

            # Plan
            if epoch % (args.il_eval_freq * 2) == 0:
                plan_info = eval_plan(args, env, agent, epoch)
            else:
                plan_info = {}

            # Logging
            logger.record_tabular('epoch', epoch)
            all_info = {}
            all_info.update(**train_infos)
            all_info.update(**skill_info)
            all_info.update(**vae_info)
            all_info.update(**plan_info)
            for key, val in all_info.items():
                logger.record_tabular(key, val)
            logger.dump_tabular()

            # Save model
            if epoch % (args.il_eval_freq * 2) == 0:
                agent.save(os.path.join(logger.get_dir(), f'agent_{epoch}.ckpt'))
    env.close()
