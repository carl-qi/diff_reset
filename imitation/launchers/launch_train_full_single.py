import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.train_full import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0915_LiftSpread_train_full_single'
    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('env_name', ['LiftSpread-v1'])
    vg.add('cached_state_path', ['datasets/0914_LiftSpread/'])
    vg.add('step_per_epoch', [500])
    vg.add('adam_iter', [3000])
    vg.add('adam_lr', [2e-1])
    vg.add('dataset_path', ['data/local/0915_LiftSpread_single/0915_LiftSpread_single_2021_09_16_20_16_44_0001/dataset.gz'])
    vg.add('il_eval_freq', [100])
    vg.add('il_num_epoch', [10000])
    vg.add('obs_noise', [0.05])
    vg.add('debug', [False])

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
