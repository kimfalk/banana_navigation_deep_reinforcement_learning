import random
import math
import numpy as np

from collections import namedtuple, deque

import torch
import torch.nn.functional as F

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward', 'done', 'priority'))

EPS = 1e-6

class ReplayMemory(object):
    """
    ReplayMemory is used to save experiences which can be sampled when training
    the neural networks.
    """

    def __init__(self,
                 action_size,
                 buffer_size,
                 batch_size,
                 seed,
                 alpha=0.6,
                 beta0=0.1,
                 beta_rate=0.99992):

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.memory = np.empty(shape=(self.buffer_size,), dtype=np.ndarray)
        self.td_errors = np.empty(shape=(self.buffer_size,), dtype=np.ndarray)

        self.n_entries = 0
        self.next_index = 0
        self.alpha = alpha
        self.beta = beta0
        self.beta_rate = beta_rate

        self.seed = seed

    def push(self, state, action, reward, next_state, done, priority):
        """
            Add experience to the memory
        """

        priority = self.td_errors[: self.n_entries].max() if self.n_entries > 0 else 1.0
        self.td_errors[self.next_index] = priority

        e = Experience(state, action, next_state, reward,  done, priority)
        self.memory[self.next_index] = e

        self.n_entries = min(self.n_entries + 1, self.buffer_size)
        self.next_index += 1
        self.next_index = self.next_index % self.buffer_size

    def sample(self, device, batch_size):
        """
            Random sample experiences
        """

        priorities = self.td_errors[:self.n_entries]

        self._update_beta()

        probs = [(priority + EPS) ** self.alpha for priority in priorities]
        probs /= np.sum(probs)

        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)

        selected_probs = np.array([probs[idx] for idx in idxs])
        weights = (self.n_entries * selected_probs) ** -self.beta
        scaled_weights = weights/weights.max()
        experiences = [self.memory[idx] for idx in idxs]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones, scaled_weights, idxs

    def __len__(self):
        return len(self.memory)

    def update(self, idxs, td_errors):
        self.td_errors[idxs] = np.abs(td_errors[:,0])

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate ** -1)
        return self.beta
