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


class Replay:
    def __init__(self, size=100000, unusual_sampling=False):
        self._replay = []
        self._size = size
        self._unusual_sampling = unusual_sampling

    def append(self, memento):
        self._replay.append(memento)
        if len(self._replay) >= self._size:
            self._replay = self._replay[-self._size:]

    def sample(self, n):
        if len(self._replay) == 0:
            return []

        if not self._unusual_sampling:
            if n > len(self._replay):
                return [self._replay[i] for i in np.random.choice(len(self._replay), size=len(self._replay), replace=False)]
            else:
                return [self._replay[i] for i in np.random.choice(len(self._replay), size=n, replace=False)]
        else:
            probs = np.array([abs(memento[2]) for memento in self._replay])
            probs = probs / np.sum(probs)
            if n > len(self._replay):
                return [self._replay[i] for i in np.random.choice(len(self._replay), size=len(self._replay), replace=False, p=probs)]
            else:
                return [self._replay[i] for i in np.random.choice(len(self._replay), size=n, replace=False, p=probs)]


class DQNEpsilonGreedyAgent(Agent):
    def __init__(self, target_switch_frames=50000, max_frames=400000,
                 eps_start=0.98, learning_rate=0.01, gamma=0.99, seed=0):
        super(Agent, self).__init__()

        self.target_switch_frames = target_switch_frames
        self.eps_start = eps_start
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
        self._replay = Replay(unusual_sampling=False)

        self._env_obs, self._env_rew, self._env_done, _ = self._env.reset()
        self._trajectory = []

    def train_step(self):
        if self.train_is_done():
            return

        if self._env_done:
            # Monte-carlc return
            value = 0.0
            for i, point in enumerate(reversed(self._trajectory)):
                value = point[2]
                self._replay.append((point[0], point[1], value))

            # Train the network
            for i in range(1):
                point_features = None
                point_actions = None
                point_returns = None
                samples = self._replay.sample(128)
                for sample in samples:
                    if point_features is None:
                        point_features = Observation.to_torch(sample[0])
                        point_actions = Variable(LongTensor([[int(sample[1])]]))
                        point_returns = Variable(FloatTensor([[sample[2]]]))
                    else:
                        point_features = t.cat((point_features, Observation.to_torch(sample[0])), 0)
                        point_actions = t.cat((point_actions, Variable(LongTensor([[int(sample[1])]]))), 0)
                        point_returns = t.cat((point_returns, Variable(FloatTensor([[sample[2]]]))), 0)

                if len(samples) > 0:
                    estimated_values = self._behavior_q.forward(point_features)
                    loss = t.mean((point_returns - t.gather(estimated_values, dim=-1, index=point_actions)) ** 2)

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

            # Reset the episode
            self._env_obs, self._env_rew, self._env_done, _ = self._env.reset()
            self._trajectory = []

        self._features = Observation.to_features(self._env_obs, self._env)

        # Select an action in an epsilon-greedy way
        action_values = self._behavior_q.forward(Observation.to_torch(self._features, volatile=True))
        action = self._env.action_space.sample()
        if np.random.rand() > self._eps:
            action = t.max(action_values, dim=-1)[1].data[0]

        # Take an action and keep current observation
        self._env_obs, self._env_rew, self._env_done, _ = self._env.step(action)
        self._features_after = Observation.to_features(self._env_obs, self._env)
        self._trajectory.append((self._features, action, self._env_rew, self._features_after))

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
        return "ddqn-eps_{}-lr_{}-gamma_{}-target_switch_{}-max_frames_{}-seed_{}" \
            .format(self.eps_start, self.learning_rate, self.gamma, self.target_switch_frames,
                    self.max_frames, self.seed)