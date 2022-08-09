import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.launchers.launch_train_full import get_dataset_path
from imitation.iclr.replay_planning import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'tune_planning'
    vg = VariantGenerator()
    vg.add('env_name', ['GatherMove-v1'])
    vg.add('dataset_path', lambda env_name: [get_dataset_path(env_name, mode, debug)])
    vg.add('seed', [100, 200])
    vg.add('adam_iter', [3000])
    vg.add('min_zlogl', [-10])
    vg.add('adam_lr', [1e-1, 1e-3])
    vg.add('resume_path', ['data/autobot/1003_LiftSpread/1003_LiftSpread/1003_LiftSpread_2021_10_03_22_25_19_0004/agent_180.ckpt'])

    vg.add('save_name', lambda min_zlogl, adam_iter, adam_lr, seed: [str(min_zlogl) + '_' + str(adam_iter) + '_' + str(adam_lr) + '_' + str(seed)])

    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(5)
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
