import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.train_full import run_task


def get_dataset_path(env_name, mode, debug):
    key = env_name + '_' + mode
    if debug:
        key = key + '_debug'
    d = {
        'LiftSpread-v1_local_debug': 'data/autobot/0923_LiftSpread_vel_loss/0923_LiftSpread_vel_loss/0923_LiftSpread_vel_loss_2021_09_24_14_26_36_0001/dataset.gz',
        'LiftSpread-v1_local': 'data/autobot/0923_LiftSpread_vel_loss/0923_LiftSpread_vel_loss/',
        'LiftSpread-v1_autobot_debug': 'data/local/0923_LiftSpread_vel_loss/0923_LiftSpread_vel_loss_2021_09_24_14_26_36_0001/dataset.gz',
        'LiftSpread-v1_autobot': 'data/local/0923_LiftSpread_vel_loss/',

        'GatherMove-v1_local_debug': 'data/seuss/0926_GatherMove_vel_loss/0926_GatherMove_vel_loss/0926_GatherMove_vel_loss_2021_09_28_11_12_14_0002/dataset.gz',
        'GatherMove-v1_local': 'data/seuss/0926_GatherMove_vel_loss/0926_GatherMove_vel_loss/',
        'GatherMove-v1_seuss_debug': 'data/local/0926_GatherMove_vel_loss/0926_GatherMove_vel_loss_2021_09_28_11_12_14_0002/dataset.gz',
        'GatherMove-v1_seuss': 'data/local/0926_GatherMove_vel_loss/',
        'GatherMove-v1_autobot': 'data/local/0926_GatherMove_vel_loss/',

        'CutRearrange-v1_local_debug': 'data/hza/buffers/buffer0/dataset0.xz',
        'CutRearrange-v1_local': 'data/hza/buffers/',
        'CutRearrange-v1_autobot_debug': 'data/hza/buffers/buffer0/dataset0.xz',
        'CutRearrange-v1_autobot': 'data/hza/buffers/',
    }
    return d[key]


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1005_cut'
    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('env_name', ['CutRearrange-v1'])
    vg.add('step_per_epoch', [100000])  # Add 500 later
    vg.add('adam_iter', [3000])
    vg.add('adam_lr', [5e-1])
    vg.add('obs_noise', [0.01])
    vg.add('pos_reset_ratio', [0.2])
    vg.add('back_prop_encoder', [True])
    vg.add('hindsight_goal_ratio', [0.5, 0.0])
    vg.add('plan_step', lambda env_name: [3] if env_name == 'CutRearrange-v1' else [2])
    # vg.add('resume_path', ['data/local/1003_cut/1003_cut_2021_10_04_14_07_40_0003/agent_160.ckpt',
    #                        'data/local/1003_cut/1003_cut_2021_10_04_14_07_40_0005/agent_340.ckpt'])
    # vg.add('adam_iter', lambda debug: [500, 3000] if not debug else [200])
    # vg.add('adam_lr', [5e-1, 2e-1])
    vg.add('debug', [debug])
    vg.add('dataset_path', lambda env_name: [get_dataset_path(env_name, mode, debug)])
    vg.add('seed', [100])
    if debug:
        vg.add('min_zlogl', [-10])
        vg.add('z_dim', [32])
        exp_prefix += '_debug'
    else:
        vg.add('min_zlogl', [-10])
        vg.add('z_dim', [16])

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
