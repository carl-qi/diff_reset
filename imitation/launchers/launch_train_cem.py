import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from plb.algorithms.cem.cem import run_task



@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--run_name', type=str, default='')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
@click.option('--env', type=str, default='RollExp-v1')
def main(mode, debug, dry, run_name, env):
    exp_prefix = '0220_cem_sweep'
    vg = VariantGenerator()
    vg.add('env_name', [env])
    vg.add('debug', [debug])
    vg.add('seed', [100])
    vg.add('run_name', [run_name])

    ########
    # to tune
    vg.add("plan_horizon", [10])
    vg.add("max_iters", [5])
    vg.add("population_size", [100])
    vg.add("num_elites", [10])
    if debug:
        exp_prefix += '_debug'

    vg.add('exp_prefix', [exp_prefix])
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
