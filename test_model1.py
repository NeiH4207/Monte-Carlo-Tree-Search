from __future__ import division

from src.environment import Environment
from src.agents import Agent
import numpy as np
import gym
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from AdasOptimizer.adasopt_pytorch import Adas
from collections import deque
from src.utils import plot


env = gym.make('CartPole-v1')
# env = gym.make('Pendulum-v0')
# env = gym.make('MountainCar-v0')

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape
A_DIM = 2
A_MAX = 1

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)
    
def test_env(args): 
    
    trainer = Agent(args, S_DIM, A_DIM, 'model')
    rewards = deque(maxlen = 100)
    reward_mean = deque(maxlen = 1000)
    for _ep in range(args.n_epochs):
        observation = env.reset()
        
        if args.test_model:
            for _iter in range(MAX_STEPS): 
                env.render()
                state = torch.from_numpy(observation).float().unsqueeze(0)
                action = torch.amax(trainer.model(state)).data.to('cpu').numpy()
                new_observation, reward, done, info = env.step(int(action))
                observation = new_observation
                if done:
                    break
        else:
            print('Training_epochs: {}'.format(_ep + 1))
            done = False
            for _iter in range(MAX_STEPS): 
                if args.show_screen:
                    env.render()
                state = torch.from_numpy(observation).float().unsqueeze(0)
                action_prob, state_value = trainer.model(state)
                act = action_prob.sample()
                log_p = action_prob.log_prob(act)
                new_observation, reward, done, info = env.step(int(act))
                
                
                if done:
                    new_observation = np.array([0, 0, 0, 0], dtype = np.float32)
                    reward = -1
                    rewards.append(_iter)
                    reward_mean.append(np.mean(rewards))
                
                # push this exp in ram
                trainer.model.store(log_p, state_value, reward, new_observation)
    
                observation = new_observation                
                if done:
                    break
                
            plot(reward_mean)
  		# perform argsimization
        trainer.learn()
        # if (_ep + 1) % 10 == 0:
        #     trainer.save_models()
        # print('Completed episodes')
"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
import argparse
from test_model1 import test_env

def get_args():
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Procon""")
    parser.add_argument("--file_name", default = "input.txt")
    parser.add_argument("--type", default = "1")
    parser.add_argument("--run", type=str, default="test_model")   
    parser.add_argument("--max_size", type=int, default= 4)   
    parser.add_argument("--min_size", type=int, default= 4)   
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=128, help="The number of state per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_super", type=float, default=0.00)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--discount", type=float, default=0.999)   
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=20000)
    parser.add_argument("--replay_memory_size", type=int, default=10000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--n_games", type=int, default=10)
    parser.add_argument("--n_maps", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=1000000)
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--test_model", type=bool, default=False)
    parser.add_argument("--chkpoint", type=str, default='./Models/model.pt')
    parser.add_argument("--show_screen", type=str, default=True)
    parser.add_argument("--load_checkpoint", type=str, default=False)
    parser.add_argument("--saved_checkpoint", type=str, default=True)   
    
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.run == "train":
        train(args)
    if args.run == "test_model":
        test_env(args)