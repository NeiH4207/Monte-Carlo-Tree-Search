from __future__ import division
import random
from copy import deepcopy as dcopy

from src.environment import Environment
from src.agents import Agent
from read_input import Data
from itertools import count
import numpy as np
from src.utils import flatten
from collections import deque
import time
from src.utils import plot

def train(args): 
    data = Data(args.min_size, args.max_size)
    env = Environment(data.get_random_map(), args.show_screen, args.max_size)
    agent = Agent(args, env.observation_dim, env.action_dim, 'agent_procon_1')  
    visual_mean_value_1 = deque(maxlen = 1000)
    visual_mean_value_2 = deque(maxlen = 1000)
    visual_value_1 = deque(maxlen = 100)
    visual_value_2 = deque(maxlen = 1000)
    
    for _ep in range(args.n_epochs):
        print('Training_epochs: {}'.format(_ep + 1))
        agent.set_environment(env.n_agents)
        
        for _game in range(args.n_games):
            done = False
            start = time.time()
            for _iter in count():
                if args.show_screen:
                    env.render()
                # initialize
                states_1, actions_1, state_values_1, log_probs_1, rewards_1 = [], [], [], [], []
                states_2, actions_2, state_values_2, log_probs_2, rewards_2 = [], [], [], [], []
                # update by step
                soft_state_1 = env.get_observation(0)
                soft_agent_pos_1 = env.get_agent_pos(0)
                
                soft_state_2 = env.get_observation(1)
                soft_agent_pos_2 = env.get_agent_pos(1)
                
                predict_actions_1 = agent.select_action_smart(soft_state_1, soft_agent_pos_1, env)
                # actions_2 = agent.select_action_smart(soft_state_2, soft_agent_pos_2, env)
                
                # fit for each agent
                for agent_id in range(env.n_agents):
                    agent_state_1 = np.array(flatten([soft_state_1, env.get_agent_state(agent_id, soft_agent_pos_1)]), dtype = np.float32)
                    if random.random() < args.lr_super:
                        action_1, log_prob_1, state_value_1 = agent.select_action_exp(agent_state_1, predict_actions_1[agent_id])
                    else:
                        action_1, log_prob_1, state_value_1 = agent.select_action(agent_state_1)
                    state_values_1.append(state_value_1)
                    valid_1, next_state_1, reward_1 = env.soft_step(agent_id, soft_state_1, action_1, soft_agent_pos_1)
                    soft_state_1 = next_state_1
                    
                    states_1.append(agent_state_1)
                    actions_1.append(action_1)
                    log_probs_1.append(log_prob_1)
                    rewards_1.append(reward_1)
                    # agent_state_2 = np.array(flatten([soft_state_2, env.get_agent_state(agent_id, soft_agent_pos_2)]))
                    # action_2, log_prob_2, state_value_2 = agent.select_action_train(agent_state_2, actions_2[agent_id])
                    # state_values_2.append(state_value_2)
                    # valid_2, next_state_2, reward_2 = env.soft_step(agent_id, soft_state_2, action_2, soft_agent_pos_2)
                    # soft_state_2 = next_state_2
                    # print(reward_1)
                    # states_2.append(agent_state_2)
                    # actions_2.append(action_2)
                    # log_probs_2.append(log_prob_2)
                    # rewards_2.append(reward_2)
                    # 
                    # actions_2.append(np.random.randint(0, agent.action_dim - 1))
                
                actions_2 = [0] * agent.n_agents
                # actions_2 = agent.select_action_smart(soft_state_2, soft_agent_pos_2, env)
                next_state, final_reward, done, _ = env.step(actions_1, actions_2, args.show_screen)
                for i in range(env.n_agents):
                    rewards_1[i] = (final_reward / env.n_agents * 0.8) + (1 - 0.8) * rewards_1[i]
                    agent.model.store(log_probs_1[i], state_values_1[i], rewards_1[i], next_state_1)
                if done:
                    break
                
            agent.learn()
            end = time.time()
            visual_value_2.append(agent.value_loss)
            visual_mean_value_1.append(np.mean(visual_value_1))
            visual_mean_value_2.append(np.mean(visual_value_2))
            if agent.steps_done % 50 == 0:
                plot(visual_mean_value_1, True, 'red')
                plot(visual_mean_value_2, True, 'blue')
                print("Time: {0: >#.3f}ms". format(1000*(end - start)))
                print("Score A/B: {}/{}". format(env.score_mine, env.score_opponent))
            if args.saved_checkpoint:
                if _game % 50 == 0:
                    agent.save_models()
            env.soft_reset()
        visual_value_1.append(env.punish)
        print('Completed episodes')
        env = Environment(data.get_random_map(), args.show_screen, args.max_size)

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
    parser.add_argument("--run", type=str, default="train")   
    parser.add_argument("--max_size", type=int, default= 6)   
    parser.add_argument("--min_size", type=int, default= 6)   
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=128, help="The number of state per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_super", type=float, default=0.00)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.01)
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
    parser.add_argument("--load_checkpoint", type=str, default=True)
    parser.add_argument("--saved_checkpoint", type=str, default=True)   
    
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.run == "train":
        train(args)
    if args.run == "test_model":
        test_env(args)