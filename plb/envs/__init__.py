import gym
from .env import PlasticineEnv
from gym import register

ENVS = [] 
register(id='RollExp-v1', # Official!!!
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=170)

register(id='RollExp-v2',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=150)

register(id='RollExp-v3',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=100)

register(id='RollExp-v4',
         entry_point=f"plb.envs.multitask_env:MultitaskPlasticineEnv",
         kwargs={'cfg_path': "roll_official.yml", "version": 1},
         max_episode_steps=50)

def make(env_name, nn=False, return_dist=False):
    env: PlasticineEnv = gym.make(env_name, nn=nn, return_dist=return_dist)
    return env
