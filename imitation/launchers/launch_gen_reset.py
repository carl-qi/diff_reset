import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.generate_reset_motion import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'gen_reset_motion'
    vg = VariantGenerator()
    if mode == 'autobot':
        vg.add('dataset_path',
               [
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0001/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0002/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0003/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0004/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0005/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0006/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0007/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0008/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0009/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0010/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0011/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0012/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0013/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0014/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0015/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0016/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0017/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0018/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0019/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0020/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0021/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0022/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0023/dataset.gz',
                   '/home/xlin3/Projects/PlasticineLab/data/local/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0024/dataset.gz'
               ])
    else:
        vg.add('dataset_path',
               ['data/autobot/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss/0916_LiftSpread_vel_loss_2021_09_16_22_56_01_0001/dataset.gz'])

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
