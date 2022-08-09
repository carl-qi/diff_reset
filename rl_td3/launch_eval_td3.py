import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from rl_td3.eval_td3 import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'eval_td3'
    vg = VariantGenerator()
    # 6 exps from ucsd, hza
    # vg.add('exp_folder', ['data/hza/rl/gathermove-0-100/0928_td3/0928_td3_2021_10_01_06_11_21_0001/',
    #                       'data/hza/rl/gathermove-0-200/0928_td3/0928_td3_2021_10_01_06_11_22_0001/',
    #                       'data/hza/rl/gathermove-1-100/0928_td3/0928_td3_2021_10_01_06_11_24_0001/',
    #                       'data/hza/rl/gathermove-1-200/0928_td3/0928_td3_2021_10_01_06_17_30_0001/',
    #                       'data/hza/rl/gathermove-2-100/0928_td3/0928_td3_2021_10_01_21_09_52_0001/',
    #                       'data/hza/rl/gathermove-2-200/0928_td3/0928_td3_2021_10_02_03_14_32_0001/'])

    # Cut-rearrange from autobot
    # vg.add('exp_folder', ['data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0001',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0002',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0003',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0004',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0005',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0006',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0007',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0008',
    #                       'data/autobot/0930_td3_cut_rearrange/0930_td3_cut_rearrange/0930_td3_cut_rearrange_2021_10_02_16_39_51_0009',])

    # particle of liftspread on seuss
    vg.add('exp_folder', [
        'data/seuss/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5_2021_09_30_15_23_23_0001/',
        'data/seuss/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5_2021_09_30_15_23_23_0002/',
        'data/seuss/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5_2021_09_30_15_23_23_0003/',
        'data/seuss/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5/0930_td3_liftspread_partial_5_2021_09_30_15_23_23_0004/'])

    # Zixuan's exp
    # vg.add('exp_folder', [
    #     'data/zixuan/0928_td3_back/0928_td3_2021_09_30_04_11_42_0001/',
    #     'data/zixuan/0928_td3_back/0928_td3_2021_09_30_04_12_09_0001/',
    #     'data/zixuan/0928_td3_back/0928_td3_2021_09_30_04_12_53_0001/',
    #     'data/zixuan/0928_td3_back/0928_td3_2021_09_30_04_13_20_0001/',
    #     'data/zixuan/0928_td3_back/0928_td3_2021_09_30_04_13_20_0001/',
    #     'data/zixuan/1001_td3_back/1001_td3_env_name=GatherMove-v1_tool_combo_id=0_seed=300_1/',
    #     'data/zixuan/1001_td3_back/1001_td3_env_name=GatherMove-v1_tool_combo_id=1_seed=300_1/',
    #     'data/zixuan/1001_td3_back/1001_td3_env_name=GatherMove-v1_tool_combo_id=2_seed=300_1/',
    # ])


    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
        # time.sleep(20)
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
            print_command=True
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
