from __future__ import division
import random
from src.environment import Environment
from src.agents import Agent
from read_input import Data
from itertools import count
import numpy as np
from collections import deque
import time
from src.utils import plot, dotdict, dtanh

cargs = dotdict({
    'run_mode': 'train',
    'visualize': True,
    'min_size': 10,
    'max_size': 20,
    'n_games': 10,
    'num_iters': 20000,
    'n_epochs': 1000000,
    'n_maps': 1000,
    'show_screen': True
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
            'saved_checkpoint': False
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
            'final_epsilon': 1e-4,
            'dir': './Models/',
            'load_checkpoint': False,
            'saved_checkpoint': False
        })
]

def train(): 
    data = Data(cargs.min_size, cargs.max_size)
    env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)
    agent = [Agent(env, args[0], 'agent_1'), Agent(env, args[1], 'agent_2')]
    wl_mean, score_mean, l_val_mean =\
        [[deque(maxlen = 10000), deque(maxlen = 10000)]  for _ in range(3)]
    wl, score, l_val = [[deque(maxlen = 100), deque(maxlen = 100)] for _ in range(3)]
    lr_super = [args[0].exp_rate, args[1].exp_rate]
    cnt_w, cnt_l = 0, 0
    beta = dtanh(env.remaining_turns)
        
    for _ep in range(cargs.n_epochs):
        print('Training_epochs: {}'.format(_ep + 1))
        for _game in range(cargs.n_games):
            done = False
            start = time.time()
            for _iter in count():
                if cargs.show_screen:
                    env.render()
                    
                """ initialize """
                actions, state_vals, log_probs, rewards, soft_state, \
                    soft_agent_pos, pred_acts = [[[], []] for i in range(7)]
                    
                """ update by step """
                for i in range(env.num_players):
                    soft_state[i] = env.get_observation(i)
                    soft_agent_pos[i] = env.get_agent_pos(i)
                    pred_acts[i] = agent[i].select_action_smart(soft_state[i], soft_agent_pos[i], env)

                """ select action for each agent """
                for agent_id in range(env.n_agents):
                    for i in range(env.num_players):
                        agent_state = env.get_states_for_step(soft_state[i])
                        agent_step = env.get_agent_for_step(agent_id)
                        act, log_p, state_val = 0, 0, 0
                        if random.random() < lr_super[i]:
                            act, log_p, state_val = agent[i].select_action_by_exp(
                                agent_state, agent_step, pred_acts[i][agent_id])
                            # if i == 1:
                            #     act = np.random.randint(0, env.n_actions - 1)
                        else:
                            act, log_p, state_val = agent[i].select_action(agent_state, agent_step)
                                
                        valid, next_state, reward =  env.soft_step(agent_id, soft_state[i], act, soft_agent_pos[i])
                        soft_state[i] = next_state
                        state_vals[i].append(state_val)
                        actions[i].append(act)
                        log_probs[i].append(log_p)
                        rewards[i].append(reward)
                # actions[1] = [np.random.randint(0, env.n_actions - 1) for _ in range(env.n_agents)]
                # actions[1] = [0] * env.n_agents
                # actions[1] = pred_acts[1]
                next_state, final_reward, done, _ = env.step(actions[0], actions[1], cargs.show_screen)
                for i in range(env.n_agents):
                    # print(rewards)
                    if done:
                        rewards[0][i] = final_reward
                        rewards[1][i] = - final_reward
                    else:
                        beta = 0.5
                        rewards[0][i] = rewards[0][i] * (1 - beta)  + beta * final_reward
                        rewards[1][i] = rewards[1][i] * (1 - beta)  - beta * final_reward
                    for j in range(env.num_players):
                        agent[j].model.store(log_probs[j][i], state_vals[j][i], rewards[j][i])
                if done:
                    score[0].append(env.players[0].total_score)
                    score[1].append(env.players[1].total_score)
                    if env.players[0].total_score > env.players[1].total_score:
                        cnt_w += 1
                    else:
                        cnt_l += 1
                    break
            agent[0].learn()
            agent[1].learn()
            end = time.time()
            l_val[0].append(agent[0].value_loss)
            l_val[1].append(agent[1].value_loss)
            wl[0].append(cnt_w)
            wl[1].append(cnt_l)
            for i in range(2):
                wl_mean[i].append(np.mean(wl[i]))
                score_mean[i].append(np.mean(score[i]))
                l_val_mean[i].append(np.mean(l_val[i]))
            
            env.soft_reset()
        if _ep % 10 == 9:
            if cargs.visualize:
                plot(wl_mean[0], False, 'red', y_title = 'num_of_wins')
                plot(wl_mean[1], True, 'blue',  y_title = 'num_of_wins')
                plot(score_mean[0], False, 'red', y_title = 'scores')
                plot(score_mean[1], True, 'blue', y_title = 'scores')
                plot(l_val_mean[0], False, 'red', y_title = 'loss_value')
                plot(l_val_mean[1], True, 'blue', y_title = 'loss_value')
                print("Time: {0: >#.3f}s". format(1000*(end - start)))
            if args[0].saved_checkpoint:
                agent[0].save_models()
            if args[1].saved_checkpoint:
                agent[1].save_models()
        # print('Completed episodes')
        # lr_super *= 0.999
        # env = Environment(data.get_random_map(), cargs.show_screen, cargs.max_size)

"""
Created on Fri Nov 27 16:00:47 2020
@author: hien
"""
if __name__ == "__main__":
    if cargs.run_mode == "train":
        train()