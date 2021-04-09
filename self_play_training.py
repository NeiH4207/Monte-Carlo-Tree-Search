from __future__ import division
import random
from src.environment import Environment
from src.agents import Agent
from read_input import Data
from itertools import count
import numpy as np
from collections import deque
from sklearn.utils import shuffle
import time
import torch
from src.utils import plot, dotdict

cargs = dotdict({
    'run_mode': 'train',
    'visualize': True,
    'min_size': 10,
    'max_size': 20,
    'n_games': 3,
    'num_iters': 20000,
    'n_epochs': 1000000,
    'n_maps': 1000,
    'show_screen': True,
})

args = [
        dotdict({
            'optimizer': 'adas',
            'lr': 1e-4,
            'exp_rate': 0.0,
            'gamma': 0.99,
            'tau': 0.01,
            'max_grad_norm': 0.3,
            'discount': 0.6,
            'num_channels': 64,
            'batch_size': 256,
            'replay_memory_size': 100000,
            'dropout': 0.6,
            'initial_epsilon': 0.1,
            'final_epsilon': 1e-4,
            'dir': './Models/',
            'load_checkpoint': False,
            'saved_checkpoint': True
        }),
        
        dotdict({
            'optimizer': 'adas',
            'lr': 1e-4,
            'exp_rate': 0.0,
            'gamma': 0.99,
            'tau': 0.01,
            'max_grad_norm': 0.3,
            'discount': 0.6,
            'batch_size': 256,
            'num_channels': 64,
            'replay_memory_size': 100000,
            'dropout': 0.4,
            'initial_epsilon': 0.1,
            'final_epsilon': 0.01,
            'dir': './Models/',
            'load_checkpoint': False,
            'saved_checkpoint': True
        })
]

def train(): 
    data = Data(cargs.min_size, cargs.max_size)
    env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
    agent = [Agent(env, args[0], 'agent_1'), Agent(env, args[1], 'agent_2')]
    
    wl_mean, score_mean, l_val_mean =\
        [[deque(maxlen = 10000), deque(maxlen = 10000)]  for _ in range(3)]
    wl, score, l_val = [[deque(maxlen = 1000), deque(maxlen = 1000)] for _ in range(3)]
    cnt_w, cnt_l = 0, 0
    temp_reward = [0] * 2
    # agent[0].model.load_state_dict(torch.load(checkpoint_path_1, map_location = agent[0].model.device))
    # agent[1].model.load_state_dict(torch.load(checkpoint_path_2, map_location = agent[1].model.device))
        
    for _ep in range(cargs.n_epochs):
        if _ep % 10 == 9:
            print('Training_epochs: {}'.format(_ep + 1))
        for _game in range(cargs.n_games):
            done = False
            start = time.time()
            current_state = env.get_observation(0)
            
            for _iter in range(env.n_turns):
                if cargs.show_screen:
                    env.render()
                    
                """ initialize """
                actions, state_vals, log_probs, rewards, soft_agent_pos = [[[], []] for i in range(5)]
                    
                """ update by step """
                for i in range(env.num_players):
                    soft_agent_pos[i] = env.get_agent_pos(i)
                    
                """ select action for each agent """
                for agent_id in range(env.n_agents):
                    for i in range(env.num_players):
                        state_step = env.get_states_for_step(current_state)
                        agent_step = env.get_agent_for_step(agent_id, soft_agent_pos)
                        act, log_p, state_val = agent[i].select_action(state_step, agent_step)
                                
                        valid, current_state, reward = env.soft_step(agent_id, current_state, act, soft_agent_pos[0])
                        current_state, soft_agent_pos = env.get_opn_observation(current_state, soft_agent_pos)
                        state_vals[i].append(state_val)
                        temp_reward[i] = reward
                        actions[i].append(act)
                        log_probs[i].append(log_p)
                    rewards[0].append(temp_reward[0] - temp_reward[1])
                    rewards[1].append(temp_reward[1] - temp_reward[0])
                # actions[1] = [np.random.randint(0, env.n_actions - 1) for _ in range(env.n_agents)]
                # actions[1] = [0] * env.n_agents
                current_state, final_reward, done, _ = env.step(actions[0], actions[1], cargs.show_screen)
                for i in range(env.n_agents):
                    for j in range(env.num_players):
                        agent[j].model.store(log_probs[j][i], state_vals[j][i], rewards[j][i])
                        
            # store the win lose battle
            in_win = env.players[0].total_score > env.players[1].total_score
            if in_win: cnt_w += 1 
            else: cnt_l += 1
                
            score[0].append(env.players[0].total_score)
            score[1].append(env.players[1].total_score)
            agent[0].learn()
            agent[1].learn()
            end = time.time()
            if _ep > 3:
                l_val[0].append(agent[0].value_loss)
                l_val[1].append(agent[1].value_loss)
                wl[0].append(cnt_w)
                wl[1].append(cnt_l)
                for i in range(2):
                    wl_mean[i].append(np.mean(wl[i]))
                    score_mean[i].append(np.mean(score[i]))
                    l_val_mean[i].append(np.mean(l_val[i]))
            
            env.soft_reset()
        if _ep % 50 == 49:
            if cargs.visualize:
                plot(wl_mean, vtype = 'Win')
                plot(score_mean, vtype = 'Score')
                plot(l_val_mean, vtype = 'Loss_Value')
                print("Time: {0: >#.3f}s". format(1000*(end - start)))
            if args[0].saved_checkpoint:
                agent[0].save_models()
                # torch.save(agent[0].model.state_dict(), checkpoint_path_1)
            if args[1].saved_checkpoint:
                agent[1].save_models()
                # torch.save(agent[1].model.state_dict(), checkpoint_path_2)
                # print('Completed episodes')
        # lr_super *= 0.999
        env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
        
if __name__ == "__main__":
     train()