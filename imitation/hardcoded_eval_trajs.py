def get_eval_traj(cached_state_path):
    """Test set"""
    if 'Roll' in cached_state_path:
        init_v = [74, 124, 10, 15, 94, 59, 24, 64, 99 ,80, 14, 80, 0, 74, 59, 14, 30, 65, 85, 100, 5, 19]
        target_v = [99, 19, 5, 20, 99, 54, 19, 84, 74, 110, 109, 100, 95, 29, 59, 64, 5, 85, 35, 20, 0, 49]
        return init_v, target_v
    else:
        raise NotImplementedError
