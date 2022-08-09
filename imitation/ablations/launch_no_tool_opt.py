import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.launchers.launch_train_full import get_dataset_path
from imitation.ablations.no_tool_opt import run_task

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1001_ablation_no_tool_opt'
    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('env_name', ['LiftSpread-v1', 'GatherMove-v1','CutRearrange-v1'])
    vg.add('step_per_epoch', [2000])  # Add 500 later
    vg.add('adam_iter', [1500])
    vg.add('adam_lr', [5e-1])
    vg.add('obs_noise', [0.01])
    vg.add('pos_reset_ratio', [0.2])
    vg.add('back_prop_encoder', [True])
    vg.add('hindsight_goal_ratio', [0.5])
    vg.add('plan_step', lambda env_name: [3] if env_name == 'CutRearrange-v1' else [2])
    vg.add('z_dim', [16])
    vg.add('debug', [debug])
    vg.add('dataset_path', lambda env_name: [get_dataset_path(env_name, mode, debug)])
    if debug:
        vg.add('min_zlogl', [-10])
        exp_prefix += '_debug'
    else:
        vg.add('min_zlogl', [-10])

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
