import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import random as rand

from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from agents.agent import Agent


class QValue(nn.Module):
    def __init__(self, n_features, n_actions=8):
        super(QValue, self).__init__()

        self.linear = nn.Linear(n_features, n_features // 2)
        self.linear1 = nn.Linear(n_features // 2, n_actions)
        
    def forward(self, features):
        '''
        input:
            features - of size (batch_size, n_features)
        output:
            action_values - estimated action values of size (batch_size, n_actions)
        '''
        return self.linear1(F.relu(self.linear(features)))


class Observation:
    @staticmethod
    def to_features(observation, env):
        look_dir = env._agent_look_dir()
        agent_pos = observation[0]
        grid = observation[2]

        feature_agent_pos = np.array([agent_pos[0], agent_pos[1]])
        feature_agent_look = np.array([look_dir[0], look_dir[1]])
        feature_grid = grid.flatten()

        return np.concatenate([feature_agent_pos, feature_agent_look, feature_grid]).reshape(1, -1)

    @staticmethod
    def num_features():
        return 204

    @staticmethod
    def to_torch(features, volatile=False):
        return Variable(FloatTensor(features), volatile=volatile)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [rand.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class DQNEpsilonGreedyAgent(Agent):
    def __init__(self, target_switch_frames=50000, max_frames=400000,
                 buffer_size=100000, eps_start=0.98, learning_rate=0.01, gamma=0.99, seed=0):
        super(Agent, self).__init__()

        self.target_switch_frames = target_switch_frames
        self.eps_start = eps_start
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_frames = max_frames
        self.seed = 0

    def train_init(self, env_definition):
        self._env = env_definition.build_env()
        self._eps = self.eps_start
        self._num_frames = 0

        # Set random seeds
        t.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        rand.seed(self.seed)

        # Q-Value approximator and optimizator
        self._behavior_q = QValue(Observation.num_features(), self._env.action_space.n)
        self._optimizer = t.optim.Adam(self._behavior_q.parameters(), lr=self.learning_rate)
        self._replay = ReplayBuffer(size=self.buffer_size)

        self._env_obs, self._env_rew, self._env_done, _ = self._env.reset()

    def train_step(self):
        if self.train_is_done():
            return

        if self._env_done:
            # Train the network
            for i in range(1):
                obs_batch, act_batch, rew_batch, _, _ = self._replay.sample(128)
                obs_batch = Observation.to_torch(obs_batch)
                act_batch = Variable(LongTensor(act_batch))
                rew_batch = Variable(FloatTensor(rew_batch))

                estimated_values = self._behavior_q.forward(obs_batch)
                estimated_values = estimated_values.view(estimated_values.size(0), -1)
                act_batch = act_batch.view(act_batch.size(0), 1)
                loss = t.mean((rew_batch - t.gather(estimated_values, dim=-1, index=act_batch)) ** 2)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            # Reset the episode
            self._env_obs, self._env_rew, self._env_done, _ = self._env.reset()

        self._features = Observation.to_features(self._env_obs, self._env)

        # Select an action in an epsilon-greedy way
        action_values = self._behavior_q.forward(Observation.to_torch(self._features, volatile=True))
        action = self._env.action_space.sample()
        if np.random.rand() > self._eps:
            action = t.max(action_values, dim=-1)[1].data[0]

        # Take an action and keep current observation
        self._env_obs, self._env_rew, self._env_done, _ = self._env.step(action)
        self._features_after = Observation.to_features(self._env_obs, self._env)
        self._replay.add(self._features, action, self._env_rew, self._features_after, self._env_done)

        # Update current observation
        self._features = self._features_after

        # Update epsilon and number of frames
        self._num_frames += 1
        self._eps = self.eps_start - (self.eps_start) * np.clip((self._num_frames / self.max_frames), 0.0, 1.0)

    def train_num_steps(self):
        return self._num_frames

    def train_is_done(self):
        return self._num_frames >= self.max_frames

    def act(self, observation):
        features = Observation.to_features(observation, self._env)
        action_values = self._behavior_q.forward(Observation.to_torch(features, volatile=True))
        action = self._env.action_space.sample()
        if np.random.rand() > self._eps:
            action = t.max(action_values, dim=-1)[1].data[0]

        return action

    def parameters(self):
        return {
            "eps_start": self.eps_start,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "target_switch_frames": self.target_switch_frames,
            "seed": self.seed,
            "max_frames": self.max_frames
        }

    def name(self):
        return "ddqn-eps_{}-lr_{}-gamma_{}-buffer_size_{}-target_switch_{}-max_frames_{}-seed_{}" \
            .format(self.eps_start, self.learning_rate, self.gamma, self.buffer_size,
                    self.target_switch_frames, self.max_frames, self.seed)