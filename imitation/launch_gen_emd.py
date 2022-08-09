import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.generate_emd import run_task
import glob
import os


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'gen_emd'
    vg = VariantGenerator()
    dataset_paths = glob.glob(os.path.join('data/hza/buffers/', '**/*dataset*'), recursive=True)
    vg.add('dataset_path', dataset_paths)
    # vg.add('dataset_path', ['data/hza/buffers/buffer10/dataset6.xz',
    #                         'data/hza/buffers/buffer10/dataset5.xz',
    #                         'data/hza/buffers/buffer10/dataset0.xz',
    #                         'data/hza/buffers/buffer10/dataset1.xz'])
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
