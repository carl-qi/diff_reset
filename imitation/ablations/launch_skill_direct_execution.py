import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.launchers.launch_train_full import get_dataset_path
from imitation.ablations.skill_direct_execution import run_task

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1001_ablation_direct_execution'
    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('env_name', ['CutRearrange-v1'])
    vg.add('plan_step', lambda env_name: [3] if env_name == 'CutRearrange-v1' else [2])
    vg.add('z_dim', [16]) # TODO should load this from the one during training
    vg.add('debug', [debug])
    vg.add('dataset_path', lambda env_name: [get_dataset_path(env_name, mode, debug)])
    # vg.add('resume_path', ['data/autobot/1003_LiftSpread/1003_LiftSpread/1003_LiftSpread_2021_10_03_22_25_19_0004/agent_180.ckpt'])#  TODO
    # vg.add('resume_path', ['data/autobot/0929_GatherMove_train_hindsight/0929_GatherMove_train_hindsight/0929_GatherMove_train_hindsight_2021_09_29_01_17_30_0014/agent_60.ckpt'])#  TODO
    vg.add('resume_path', ['data/autobot/1003_cut/1003_cut/1003_cut_2021_10_04_14_07_40_0003/agent_160.ckpt'])
    if debug:
        vg.add('min_zlogl', [-10])
        exp_prefix += '_debug'
    else:
        vg.add('min_zlogl', [-5])

    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
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
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
