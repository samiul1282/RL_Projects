# %% 
import gymnasium as gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("parking-v0")


# %% 

n_sampled_goal = 4 

# SAC hyperparams: 
model = SAC(
    "MultiInputPolicy", 
    env, 
    replay_buffer_class=  HerReplayBuffer, 
    replay_buffer_kwargs= dict(
        n_sampled_goal = n_sampled_goal, 
        goal_selection_strategy = "future", 
    ), 
    verbose= 1, 
    buffer_size= int(1e6), 
    learning_rate= 1e-3, 
    gamma= 0.95, 
    batch_size = 256, 
    policy_kwargs=dict(net_arch = [256, 256, 256])
)

model.learn(int(2e5))
model.save("her_sac_highway")





