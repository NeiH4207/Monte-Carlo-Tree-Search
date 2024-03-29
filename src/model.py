import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical
from AdasOptimizer.adasopt_pytorch import Adas
from torch.optim import Adam, SGD
from collections import deque

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)
    
class Policy(nn.Module):
    def __init__(self, env, args, chkpoint_file = 'Models/'):
        # game params
        self.board_x, self.board_y = env.get_ub_board_size()
        self.action_size = env.n_actions
        self.n_inputs = env.n_inputs
        self.lr = args.lr
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(self.n_inputs, args.num_channels, 3, stride=1, padding=1).to(self.device)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1).to(self.device)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1).to(self.device)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1).to(self.device)

        self.bn1 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn2 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn3 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.bn4 = nn.BatchNorm2d(args.num_channels).to(self.device)
        self.fc1 = nn.Linear(args.num_channels*(self.board_x - 4)*(self.board_y - 4) \
                             + env.agent_step_dim, 1024).to(self.device)
        self.fc_bn1 = nn.BatchNorm1d(1024).to(self.device)

        self.fc2 = nn.Linear(1024, 512).to(self.device)
        self.fc_bn2 = nn.BatchNorm1d(512).to(self.device)

        self.fc3 = nn.Linear(512, self.action_size).to(self.device)

        self.fc4 = nn.Linear(512, 1).to(self.device)
        
        self.entropies = 0
        self.action_probs = [[], []]
        self.state_values = [[], []]
        self.rewards = [[], []]
        self.next_states = [[], []]
        if args.optimizer == 'adas':
            self.optimizer = Adas(self.parameters(), lr=self.lr)
        elif args.optimizer == 'adam':
            self.optimizer = Adam(self.parameters(), lr=self.lr)
        else:
            self.optimizer = SGD(self.parameters(), lr=self.lr)

    def forward(self, s, agent):
        #                                                           s: batch_size x n_inputs x board_x x board_y
        s = s.view(-1, self.n_inputs, self.board_x, self.board_y)    # batch_size x n_inputs x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))
        s = torch.cat((s,agent),dim=1)
        s = F.dropout(F.relu(self.fc1(s)), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc2(s)), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return pi, F.softmax(pi, dim=1), v # torch.tanh(v)
    
    def step(self, obs, agent):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = torch.from_numpy(obs).to(self.device)
        agent = torch.from_numpy(agent).to(self.device)
        _, pi, v = self.forward(obs, agent)

        return pi.detach().to('cpu').numpy(), v.detach().to('cpu').numpy()

    def store(self, player_ID, prob, state_value, reward):
        self.action_probs[player_ID].append(prob)
        self.state_values[player_ID].append(state_value)
        self.rewards[player_ID].append(reward)
    
    def clear(self):
        self.action_probs = [[], []]
        self.state_values = [[], []]
        self.rewards = [[], []]
        self.next_states = [[], []]
        self.entropies = 0
    
    def get_data(self):
        return self.action_probs, self.state_values, self.rewards
        
    def optimize(self):
        self.optimizer.step()
        
    def reset_grad(self):
        self.optimizer.zero_grad()

    def save_checkpoint(self, name):
        # print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.args.dir + name + '.pt')

    def load_checkpoint(self, name):
        # print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.args.dir + name + '.pt', map_location = self.device))
        