from stable_baselines3 import PPO 
from stable_baselines3.common.envs import SimpleMultiObsEnv 

env = SimpleMultiObsEnv(random_start= False)

model = PPO("MultiInputPolicy", env, verbose= 1)
model.learn(total_timesteps= 100_000)
