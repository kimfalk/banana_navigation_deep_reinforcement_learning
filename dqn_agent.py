import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import DQN
from replay_memory import ReplayMemory

LR = 5e-4
BUFFER_SIZE = 10000     # replay memory size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3                # for soft update of target parameters
UPDATE_EVERY = 3        # how often should the network be updated.
EPS = 1e-6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DqnAgent():

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 replay_buffer: ReplayMemory,
                 seed: int,
                 batch_size=BATCH_SIZE,
                 update_every=UPDATE_EVERY,
                 tau=TAU,
                 gamma=GAMMA):
        """Initialize the agent"""

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.tau = tau
        self.update_target_every = update_every
        self.gamma = gamma

        self.qnet_local = DQN(state_size, action_size, seed).to(device)
        self.qnet_target = DQN(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=LR)
        self.max_gradient_norm = float('inf')

        self.memory = replay_buffer

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        self.memory.push(state,
                         action,
                         reward,
                         next_state,
                         done)

        self.t_step = (self.t_step + 1) % self.update_target_every
        if self.t_step == 0:

            if self.memory.n_entries > BATCH_SIZE*4:
                # Only learn if replay memory is big enough
                experiences = self.memory.sample(device, self.batch_size)
                self.learn(experiences, self.gamma)

                # Update the target network
                self.soft_update(self.qnet_local, self.qnet_target, TAU)

    def act(self, state, epsilon=0.1):
        """
            Given a state responds with the best action according to the current policy
        :param state:  current state
        :param epsilon:    epsilon decides the chance of returning a random action
        :return:       action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        np_action_values = action_values.cpu().data.numpy()
        if random.random() > epsilon:
            return np.argmax(np_action_values), np_action_values
        else:
            return random.choice(np.arange(self.action_size)), np_action_values

    def learn(self, experiences, gamma):
        """
        learn/optimize the model using the sample experiences
        """

        states, actions, rewards, next_states, dones, weights, idxs = experiences

        # P.E.R: weights = torch.from_numpy(weights).float().to(device)

        local_max_action = self.qnet_local(next_states).max(1, keepdim=True)[1]

        # Using the online network to find next action value (predicted by target network)
        target_next = self.qnet_target(next_states.float()).gather(1, local_max_action)

        # target Q values
        q_targets = rewards + (gamma * target_next * (1-dones))

        # Q(s_t, a)
        q_expected = self.qnet_local(states).gather(1, actions)

        # Huber loss
        # P.E.R: td_errors = q_expected - q_targets
        # P.E.R: value_loss = torch.where(td_errors < 1, (td_errors.pow(2).mul(0.5)), td_errors - 0.5)

        value_loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet_local.parameters(),
                                        self.max_gradient_norm)
        self.optimizer.step()

        # P.E.R
        #td_errors = np.abs(td_errors.detach().cpu().numpy())
        #self.memory.update(idxs, td_errors)

    def soft_update(self, local_model, target_model, tau):

        assert tau < 1

        for target, local in zip(target_model.parameters(),
                                 local_model.parameters()):

            target.data.copy_(tau * local.data + (1.0 - tau) * target.data)



if __name__ == '__main__':

    ### The following was used to train the agent outside of the notebook.

    from unityagents import UnityEnvironment
    import numpy as np
    import random
    import torch
    import numpy as np
    from collections import deque
    import time

    env = UnityEnvironment(file_name="data/Banana")

    SEED = 42
    state_size = 37
    action_size = 4

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    def dqn(n_episodes=2000,
            max_t=1000,
            eps_start=1.0,
            eps_end=0.1,
            eps_decay=0.99,
            alpha=0.6,
            beta0=0.1,
            beta_rate=0.99992):
        """
        Deep Q-Network learning
        """
        agent = DqnAgent(state_size,
                         action_size,
                         ReplayMemory(action_size, BUFFER_SIZE, BATCH_SIZE, SEED, alpha, beta0, beta_rate),
                         SEED)
        scores = []
        scores_window = deque(maxlen=100)
        epsilon = eps_start

        for i_episode in range(1, n_episodes + 1):

            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            action_values = np.zeros(1)

            for t in range(max_t):

                action, action_values = agent.act(state, epsilon)

                env_info = env.step(action)[brain_name]       # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]                  # get the reward
                done = env_info.local_done[0]                 # get the done status

                agent.step(state,
                           action,
                           reward,
                           next_state,
                           done)                              # inform agent about what happened

                state = next_state
                score += reward

                if done:
                    break

            scores_window.append(score)
            scores.append(score)

            epsilon = max(eps_end, epsilon * eps_decay)

            log_str = '\rEpisode {}\tAverage Score: {:.2f} \tbeta: {:.2f}\taction_value: {:.2f}'
            print(log_str.format(i_episode, np.mean(scores_window), agent.memory.beta, action_values.mean()), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                             np.mean(scores_window)))
                torch.save(agent.qnet_local.state_dict(), 'checkpoint.pth')
                break
        return scores


    scores = dqn()
