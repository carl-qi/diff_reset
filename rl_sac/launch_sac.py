import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from rl_sac.run_sac import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
@click.option('--run_name', type=str, default='')
def main(mode, debug, dry, run_name):
    exp_prefix = '0217_sac_sweep'
    vg = VariantGenerator()
    vg.add('algo', ['SAC'])
    vg.add('env_name', ['RollExp-v1'])
    vg.add('tool_combo_id', [0])  # Binary encoding of what tools to use
    vg.add('emd_downsample_num', [500])
    vg.add('seed', [188, 288, 388, 100])
    vg.add('lr', [3e-4])
    vg.add('max_timesteps', [10000000])
    vg.add('replay_k', [0])  # replay_k = 0 means no hindsight relabeling, which should be much faster.
    vg.add('train_freq', [170])  # Train after collecting each episode. This will make overall training faster.
    vg.add('resume_path', [None])
    vg.add('batch_size', [10])
    vg.add('buffer_horizon', [170])
    vg.add('run_name', [run_name])
    if debug:
        exp_prefix += '_debug'
        vg.add('num_env', [1])
        vg.add('start_timesteps', [5000])
        vg.add('eval_freq', [50])
    else:
        vg.add('num_env', [1])
        vg.add('start_timesteps', [5000])
        vg.add('eval_freq', [50])
    vg.add('exp_prefix', [exp_prefix])
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
