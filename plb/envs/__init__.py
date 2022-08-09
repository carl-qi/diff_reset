import gym
from .env import PlasticineEnv
from gym import register

ENVS = []

register(id='Roll-v3', # roll to 2d 150
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll.yml", "version": 1},
         max_episode_steps=150)

register(id='Roll-v2', # roll to 2d 100
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll.yml", "version": 1},
         max_episode_steps=100)

register(id='Roll-v1', # roll to 2d short horizon
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll.yml", "version": 1},
         max_episode_steps=50)

register(id='Roll-v0', # roll to 1d
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_1102.yml", "version": 1},
         max_episode_steps=50)

register(id='RollLong-v1', # roll to 2d long
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll.yml", "version": 1},
         max_episode_steps=170)

register(id='RollTest-v1', # roll to 2d long, red rolling pin
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_test.yml", "version": 1},
         max_episode_steps=170)

register(id='RollDev-v1',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_dev.yml", "version": 1},
         max_episode_steps=170)
        
register(id='RollTest-v2', # test with new action space of rolling pin
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_test2.yml", "version": 1},
         max_episode_steps=170)

register(id='RollExp-v1', # test with new action space of rolling pin
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=170)

register(id='RollExp-v2', # test with new action space of rolling pin
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=150)

register(id='RollExp-v3', # test with new action space of rolling pin
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=100)

register(id='RollExp-v4', # test with new action space of rolling pin
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=50)

register(id='RollTest-v3', # test with new action space of rolling pin, action space = tool flow
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_test3.yml", "version": 1},
         max_episode_steps=170)

register(id='RollTestShort-v2', # roll to 2d long, red rolling pin
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_test2.yml", "version": 1},
         max_episode_steps=110)

def make(env_name, nn=False, return_dist=False):
    env: PlasticineEnv = gym.make(env_name, nn=nn, return_dist=return_dist)
    return env
