import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.gen_data import run_task
import os

vel_loss_weights = {
    'LiftSpread-v1': 0.02,
    'GatherMove-v1': 0.05,
    'Roll-v1': 0.0,
    'Roll-v2': 0.0,
    'Roll-v3': 0.0,
    'RollLong-v1':0.0,
    'RollTest-v1':0.0,
    'RollTest-v2':0.0,
    'RollDev-v1':0.0,
    'RollTestShort-v2': 0.0,
    'RollExp-v1': 0.0,
    'RollExp-v2': 0.0,
    'RollExp-v3': 0.,
    'RollExp-v4': 0.,
}

episode_length = {
    'Roll-v1': 50,
    'Roll-v2': 100,
    'Roll-v3': 150,
    'RollLong-v1':170,
    'RollTest-v1':170,
    'RollTest-v2':170,
    'RollDev-v1':170,
    'RollTestShort-v2': 110,
    'RollExp-v1': 170,
    'RollExp-v2': 150,
    'RollExp-v3': 100,
    'RollExp-v4': 50,
}


@click.command()
@click.argument('mode', type=str, default='local') # suess
@click.option('--env', type=str, default='RollExp-v1')
@click.option('--run_name', type=str, default='')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
@click.option('--opt_mode', type=str, default='multireset')
@click.option('--num_moves', type=int, default=1)
@click.option('--seed', type=int, default=0)
def main(mode, env, debug, dry, run_name, opt_mode, num_moves, seed):
    # exp_prefix = '1129_Roll_exp_gendemo'

    exp_prefix = '0515_multireset_init'

    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('env_name', [env])
    vg.add('eps_length', [episode_length[env]])
    vg.add('seed', [seed])
    vg.add('reset_loss_freq', [50 if opt_mode=='twopass' else 9999])
    vg.add('reset_loss_steps', [50 if opt_mode=='twopass' else 0])
    vg.add('reset_lr', [0.05])
    vg.add('num_moves', [num_moves])
    vg.add('reset_primitive', [num_moves > 1])
    vg.add('action_init', ['zero'])
    vg.add('debug_gradient', [False])
    vg.add('opt_mode', [opt_mode])
    vg.add('run_name', [run_name])
    vg.add('exp_prefix', [exp_prefix])
    vg.add('data_name', ['eval'])  # demo, single
    vg.add('num_resets', [2])
    vg.add('buffer_horizon', [150])
    vg.add('lr', [0.005])

    vg.add('gen_num_batch', lambda data_name: [5])  # Split the data generation task to 20 machines

    vg.add('gen_batch_id', lambda gen_num_batch: list(range(gen_num_batch)))
    vg.add('vel_loss_weight', lambda env_name: [vel_loss_weights[env_name]])

    vg.add('wandb_group', [exp_prefix])
    vg.add('use_wandb', [not debug])
    if debug:
        exp_prefix += '_debug'
        vg.add('num_trajs', [10])
        vg.add('gd_max_iter', [200])
    else:
        vg.add('num_trajs', [10])
        vg.add('gd_max_iter', [1000])
        # vg.add('gd_max_iter2', [300])
        
    print(vg.variants())
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
