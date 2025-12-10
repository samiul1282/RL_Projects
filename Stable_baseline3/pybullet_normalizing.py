from pathlib import Path 

import pybullet_envs_gymnasium 

from stable_baselines3.common.vec_env import VecNormalize 
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3 import PPO 


# alter
vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs= 1) 

# normalize 
vec_env = VecNormalize(vec_env, norm_obs= True, norm_reward= True, 
                       clip_obs= 10.0)


model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps= 2_000) 

# save the info 
log_dir = Path("./tmp/")
model.save(log_dir/"ppo_halfcheetah")
stats_path = log_dir/"vec_normalize.pkl"
vec_env.save(stats_path)


del model, vec_env 

# Load the saved statistics 
vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs= 1)
vec_env = VecNormalize.load(stats_path, vec_env) 

vec_env.training = False 
vec_env.norm_reward = False 

# load the agent  
model = PPO.load(log_dir/"ppo_halfcheetah", env = vec_env)


