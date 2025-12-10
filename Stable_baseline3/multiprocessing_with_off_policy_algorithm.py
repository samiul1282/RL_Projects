import gymnasium as gym 
from stable_baselines3 import SAC 
from stable_baselines3.common.env_util import make_vec_env 




vec_env = make_vec_env("Pendulum-v1", n_envs= 4, seed = 0)


model = SAC("MlpPolicy", vec_env, train_freq= 1, gradient_steps= 2, verbose= 1)
model.learn(total_timesteps= 10_000)


# create a seperate environment for rendering/visual evaluation 
eval_env = gym.make("Pendulum-v1", render_mode = "human")

obs, _ = eval_env.reset()

for _ in range(1000):
    action, states = model.predict(obs, deterministic= True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    eval_env.render()

    