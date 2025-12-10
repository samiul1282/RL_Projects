import gymnasium as gym 
from lib import dqn_model 
from lib import wrappers 


from dataclasses import dataclass 
import argparse 
import time 
import numpy as np 
import collections 
import typing as tt 

import torch 
import torch.nn as nn 
import torch.optim as optim 

from torch.utils.tensorboard.writer import SummaryWriter 


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19 

GAMMA = 0.99 
BATCH_SIZE = 32 
REPLAY_SIZE = 10,000 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default= "cpu", help = "Device name, default = cpu")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space,shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment = "-"+args.env)
    print(net)


    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START 

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None



