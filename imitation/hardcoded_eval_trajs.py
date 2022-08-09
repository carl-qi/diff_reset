def get_eval_traj(cached_state_path):
    """Test set"""
    if '0820_PushSpread' in cached_state_path:
        # Push only
        init_v_1 = [82, 75, 74, 81, 96]
        target_v_1 = [22, 35, 14, 51, 26]

        # Spread only
        init_v_2 = [46, 55, 74, 89, 92]
        target_v_2 = [142, 155, 174, 189, 196]

        # Push + Spread
        init_v_3 = [45, 57, 59, 67, 73, 79, 83, 86, 94, 97]
        target_v_3 = [101, 105, 106, 113, 117, 119, 122, 126, 129, 130]

        init_v = init_v_1 + init_v_2 + init_v_3
        target_v = target_v_1 + target_v_2 + target_v_3
        return init_v[::4], target_v[::4]  # Smaller dataset for faster evaluation
    elif '0923_LiftSpread' in cached_state_path:
        # Small
        # init_v = [106, 36]
        # target_v = [36, 117]
        # Larger (Use five for faster evaluation), all require two stages
        init_v_1 = [104, 104, 106, 106, 109]
        target_v_1 = [109, 112, 143, 166, 185]
        return init_v_1, target_v_1
    elif '0926_GatherMove' in cached_state_path:
        init_v = [0, 2, 4, 8, 16]
        target_v = [102, 103, 102, 103, 104]
        return init_v, target_v
    elif 'CutRearrange' in cached_state_path:
        # Correct one
        # init_v = [24, 27, 33, 42, 51]
        # target_v = [26, 29, 35, 44, 53]
        # Only one rearrange for ICLR
        init_v = [24, 27, 33, 42, 51]
        target_v = [25, 28, 34, 43, 52]
        return init_v, target_v
    elif 'Roll' in cached_state_path:
        init_v = [74, 124, 10, 15, 94, 59, 24, 64, 99 ,80, 14, 80, 0, 74, 59, 14, 30, 65, 85, 100, 5, 19]
        target_v = [99, 19, 5, 20, 99, 54, 19, 84, 74, 110, 109, 100, 95, 29, 59, 64, 5, 85, 35, 20, 0, 49]
        return init_v, target_v
    else:
        raise NotImplementedError
