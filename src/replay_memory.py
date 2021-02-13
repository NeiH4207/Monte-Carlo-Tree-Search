import numpy as np
import random
from collections import deque
import torch


class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.maxSize = max_size
        self.len = 0

    def store_transition(self, state, log_prob, reward, next_state):
        """
		adds a particular transaction in the memory buffer
        """
        reward = torch.FloatTensor([reward])
        transition = (state, log_prob, reward, next_state)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        samples a random batch from the replay memory buffer
        :param count: batch size
        :return: batch (numpy array)
        """
        batch = []
        batch_size = min(batch_size, self.len)
        batch = random.sample(self.buffer, batch_size)
        s_arr = [arr[0] for arr in batch]
        s_arr = torch.stack(s_arr)
        log_p_arr = [arr[1] for arr in batch]
        log_p_arr = torch.stack(log_p_arr)
        r_arr = [arr[2] for arr in batch]
        r_arr = torch.stack(r_arr)
        ns_arr = [arr[3] for arr in batch]
        ns_arr = torch.stack(ns_arr)
        return s_arr, log_p_arr, r_arr, ns_arr