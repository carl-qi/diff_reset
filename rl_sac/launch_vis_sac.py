import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from rl_sac.visualize_sac import run_task


def get_sac_agent_path(env_name, tool_id):
    name = f'{env_name}-{tool_id}'
    d = {
        'LiftSpread-v1-0': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0001/', 29900],
        'LiftSpread-v1-1': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0004/', 25000],
        'LiftSpread-v1-2': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0005/', 6850],
        'GatherMove-v1-0': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0008/', 1900],
        'GatherMove-v1-1': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0010/', 17100],
        'GatherMove-v1-2': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0011/', 5400],
        'CutRearrange-v1-0': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0013/', 10850],
        'CutRearrange-v1-1': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0016/', 2500],
        'CutRearrange-v1-2': [
            'data/seuss/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110/rebuttal_sac_multistage_1110_2021_11_10_23_53_56_0017/', 9550]}
    return d[name]


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'visualize_rl'
    vg = VariantGenerator()
    vg.add('num_env', [1])

    task = 'demo' # 'eval' : Save RL trajectory, 'demo' : save hd videos
    if task == 'eval':
        vg.add('task', ['eval'])
        vg.add('env_name', ['LiftSpread-v1', 'GatherMove-v1', 'CutRearrange-v1'])
        vg.add('tool_combo_id', [0, 1, 2])  # Binary encoding of what tools to use
        # vg.add('env_name', ['CutRearrange-v1'])
        # vg.add('tool_combo_id', [1, 2])  # Binary encoding of what tools to use
        vg.add('sac_agent_path', lambda env_name, tool_combo_id: [get_sac_agent_path(env_name, tool_combo_id)])
    elif task == 'demo':
        vg.add('task', ['demo'])  # 'eval' : Save RL trajectory, 'demo' : save hd videos
        from glob import glob
        vg.add('traj_folder', glob('data/local/visualize_rl/visualize_rl_2021_11_15_00_28_20_0001'))
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
