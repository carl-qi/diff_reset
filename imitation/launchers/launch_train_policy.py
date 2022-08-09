import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0802_PushSpread_train_policy'
    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('env_name', ['PushSpread-v1'])
    vg.add('chamfer_loss', [0.])
    vg.add('task', ['train_policy'])
    vg.add('il_num_epoch', [5000])
    vg.add('batch_size', [128])
    vg.add('hindsight_goal_ratio', [0.])  # Percentage of hindsight goal
    vg.add('frame_stack', [1, 2])
    vg.add('img_mode', ['rgbd'])
    vg.add('il_lr', [3e-4])
    if mode == 'local':
        if debug:
            vg.add('dataset_path',
                   ['data/autobot/0727_PushSpread_lr_0.1/0727_PushSpread_lr_0.1/0727_PushSpread_lr_0.1_2021_07_28_18_14_06_0003/dataset.gz'])
            vg.add('il_eval_freq', [20])
            vg.add('debug_overfit_test', [True])
            vg.add('debug_overfit_traj_ids', [[64], [64, 65, 66, 67], [8, 13, 38, 53, 72, 86, 102, 103],
                                              [8, 13, 38, 53, 72, 86, 102, 103, 9, 14, 39, 54, 73, 87, 104, 105]])
            vg.add('obs_noise', [0.2])
            # vg.add('debug_overfit_traj_ids', [[86]])
        else:
            vg.add('dataset_path',
                   ['data/autobot/0727_PushSpread_lr_0.1/0727_PushSpread_lr_0.1/0727_PushSpread_lr_0.1_2021_07_28_18_14_06_0003/dataset.gz'])
            vg.add('il_eval_freq', [20])
            vg.add('debug_overfit_test', [True])
            vg.add('debug_overfit_traj_ids', [[64], [64, 65, 66, 67]])
            vg.add('obs_noise', [0.1, 0.05])
            # vg.add('debug_overfit_traj_ids', [[86]])
    elif mode == 'autobot':
        vg.add('dataset_path', ['data/local/0727_PushSpread_lr_0.1'])
        vg.add('il_eval_freq', [100])
        vg.add('obs_noise', [0.25, 0.3])
    else:
        raise NotImplementedError
    if debug:
        exp_prefix += '_debug'
        vg.add('gd_num_steps', [50])
    else:
        pass

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
