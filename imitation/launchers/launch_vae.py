import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0729_vae'
    vg = VariantGenerator()
    if mode =='local':
        vg.add('dataset_path', ['data/autobot/0710_PushSpread/0710_PushSpread/dataset.gz'])
    else:
        vg.add('dataset_path', ['data/local/0727_PushSpread_lr_0.1'])
    vg.add('encoder_lr', [1e-3, 3e-4, 1e-4])
    vg.add('task', ['train_encoder'])
    vg.add('encoder_beta', [1., 10., 50.])
    vg.add('num_epoch', [1000])
    vg.add('batch_size', [128])

    if not debug:
        pass
    else:
        exp_prefix += '_debug'

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
