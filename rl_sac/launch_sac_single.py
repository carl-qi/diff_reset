import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from rl_sac.run_sac import run_task


def get_tool_combo_id(env_name):
    d = {'Spread-v1': 1, 'Lift-v1': 0}
    return d[env_name]


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'rebuttal_sac_single_stage_liftspread_1111'
    vg = VariantGenerator()
    vg.add('algo', ['SAC'])
    vg.add('env_name', ['Spread-v1'])
    vg.add('tool_combo_id', lambda env_name: [get_tool_combo_id(env_name)])
    vg.add('emd_downsample_num', [500])
    vg.add('seed', [400, 500, 600])
    vg.add('max_timesteps', [10000000])
    vg.add('replay_k', [0])  # replay_k = 0 means no hindsight relabeling, which should be much faster.
    vg.add('train_freq', lambda num_env: [50 * num_env])  # Train after collecting each episode. This will make overall training faster.
    vg.add('resume_path', [None])
    if debug:
        exp_prefix += '_debug'
        vg.add('num_env', [1])
        vg.add('start_timesteps', [50])
        vg.add('eval_freq', [1])
    else:
        vg.add('num_env', [2])
        vg.add('start_timesteps', [5000])
        vg.add('eval_freq', [50])
    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
        # time.sleep(20)
        compile_script = wait_compile = None

        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            print_command=True
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
