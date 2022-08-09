import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from rl_sac.visualize_sac import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'visualize_diffskill_single'
    vg = VariantGenerator()
    vg.add('num_env', [1])

    task = 'demo'  # 'eval' : Save RL trajectory, 'demo' : save hd videos
    if task == 'eval':
        vg.add('task', ['eval'])
        vg.add('env_name', ['Spread-v1'])
        # vg.add('sac_agent_path', lambda env_name: [get_sac_agent_path(env_name)])
    elif task == 'demo':
        vg.add('task', ['demo'])  # 'eval' : Save RL trajectory, 'demo' : save hd videos
        from glob import glob
        vg.add('traj_folder', ['data/local/diffskill_single_stage/diffskill_single_stage_2021_11_18_18_19_00_0001/'])
        vg.add('img_size', [256])

    if debug:
        exp_prefix += '_debug'
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
