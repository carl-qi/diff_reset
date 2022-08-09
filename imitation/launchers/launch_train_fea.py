import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0807_PushSpread_fea'
    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('env_name', ['PushSpread-v1'])
    vg.add('chamfer_loss', [0.])
    vg.add('task', ['train_feas'])

    vg.add('num_epoch', [100])
    vg.add('batch_size', [128])

    vg.add('dataset_path', [None])
    vg.add('gen_agent_dataset', [False])
    vg.add('gen_num_batch', lambda gen_agent_dataset: [16] if gen_agent_dataset else [1])  # Split the data generation task to 20 machines
    vg.add('gen_batch_id', lambda gen_num_batch: list(range(gen_num_batch)))
    vg.add('positive_aug', [False])
    vg.add('bin_succ', lambda gen_agent_dataset: [False] if not gen_agent_dataset else [True])
    vg.add('soft_clipping', [True])
    if debug:
        assert mode == 'local'
        vg.add('num_trajs', [64])
        vg.add('il_eval_freq', [1])
        vg.add('agent_dataset_path', [
            'data/autobot/0803_PushSpread_fea/0803_PushSpread_fea/0803_PushSpread_fea_2021_08_03_11_32_34_0001/agent_dataset.gz'])
        vg.add('agent_path', [
            'data/autobot/0802_PushSpread_train_policy/0802_PushSpread_train_policy/0802_PushSpread_train_policy_2021_08_02_15_28_30_0001/agent_1800.ckpt'])
        exp_prefix += '_debug'
    else:
        assert mode == 'autobot'
        vg.add('il_eval_freq', [5])
        vg.add('num_trajs', [1000])
        # vg.add('agent_dataset_path', ['./data/autobot/0718_PushSpread_train_fea/0718_PushSpread_train_fea/'])
        vg.add('agent_dataset_path', ['data/local/0803_PushSpread_fea/'])
        vg.add('agent_path', [
            'data/local/0802_PushSpread_train_policy/0802_PushSpread_train_policy_2021_08_02_15_28_30_0001/agent_1800.ckpt'])

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
