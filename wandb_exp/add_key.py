import wandb

def group_runs_with_legend(project_name, group_name, custom_legend, group_key='custom_key_1'):
    """
    Example:
    def custom_legend(config):
        return f"Env name: {config['env_name']}, seed: {config['seed']}"
    group_runs('DiffSkill', '1225_liftspread_train_full', custom_legend)

    :param project_name:
    :param group_name: Group of the experiments
    :param custom_legend: A function that takes in config and return the legend
    :param group_key: Key that will show up in the wandb terminal
    :return:
    """

    api = wandb.Api()
    runs = api.runs(path=project_name,  # Project name
                    filters={"config.wandb_group": group_name})
    for run in runs:
        run.config[group_key] = custom_legend(run.config)
        print(f"Set exp {run.config['exp_name']}, key: {group_key}, val:{run.config[group_key]}")
        run.update()

def default_legend(config):
    return f"Env name: {config['env_name']}"


def run_cached_grouping():
    # group_runs_with_legend('DynAbs', 'test_z4', lambda config: f"actor_reward_dim: {config['actor_latent_dim']}, relative: {config['t_relative']}",
    #                        args.key)
    # group_runs_with_legend('DynAbs', '0415_test_z4',
    #                        lambda config: f"dimactor: {config['actor_latent_dim']}, dimfea: {config['fea_latent_dim']}, relative: {config['t_relative']}",
    #                        args.key)

    # group_runs_with_legend('DynAbs', '0415_test_z4',
    #                        lambda config: f"dimactor: {config['actor_latent_dim']}, dimfea: {config['fea_latent_dim']}",args.key)
    #group_runs_with_legend('DynAbs', '0415_cached_test_z4',
    #                       lambda config: f"dimactor: {config['actor_latent_dim']}, dimfea: {config['fea_latent_dim']}", args.key)
    group_runs_with_legend('Dough', '0515_multireset_init',
                      lambda config: f"num_resets: {config['num_resets']}, horizon: {config['buffer_horizon']}", args.key)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_group', type=str, default=None)
    parser.add_argument('--key', type=str, default='custom_key_1')

    args = parser.parse_args()

    if args.exp_group is None:
        run_cached_grouping()
    else:
        group_runs_with_legend('DynAbs', args.exp_group, default_legend, args.key)
