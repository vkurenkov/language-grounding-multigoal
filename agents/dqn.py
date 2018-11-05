import torch.nn as nn
import torch.nn.functional as F
import torch as t
import numpy as np
import random as rand
import copy
import tensorboardX as tensorboard

from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from agents.agent import Agent


class QValueConv(nn.Module):
    def __init__(self, n_features, n_actions=8):
        super(QValueConv, self).__init__() 

        self.linear = nn.Conv2d(3, 3, 2)
        linear_n_features = self._out_size(self.linear, 10, 10)
        
        self.linear1 = nn.Conv2d(3, 3, 2)
        linear1_n_features = self._out_size(self.linear1, linear_n_features[0], linear_n_features[1])
        tot_features = linear1_n_features[0] * linear1_n_features[1] * linear1_n_features[2]

        self.linear2 = nn.Linear(tot_features, n_actions)
        
    def _out_size(self, conv, width, height):
        h_out = np.floor((height+2*conv.padding[0]-conv.dilation[0]*(conv.kernel_size[0]-1)-1)/conv.stride[0]+1)
        w_out = np.floor((width+2*conv.padding[1]-conv.dilation[1]*(conv.kernel_size[1]-1)-1)/conv.stride[1]+1)
        return int(w_out), int(h_out), int(conv.out_channels)

    def forward(self, features):
        '''
        input:
            features - of size (batch_size, n_features)
        output:
            action_values - estimated action values of size (batch_size, n_actions)
        '''
        temp = F.relu(self.linear1(F.relu(self.linear(features))))
        temp = temp.view(features.size(0), -1)
        return self.linear2(temp)

class QValueLinear(nn.Module):
    def __init__(self, n_features, n_actions=8):
        super(QValueLinear, self).__init__()

        self.values = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_actions)
        )
        
    def forward(self, features):
        '''
        input:
            features - of size (batch_size, n_features)
        output:
            action_values - estimated action values of size (batch_size, n_actions)
        '''
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        
        return self.values(features)


class Observation:
    @staticmethod
    def to_features(observation, env):
        look_dir = env._agent_look_dir()
        agent_pos = observation[0]
        look_pos = (agent_pos[0] + look_dir[0], agent_pos[1] + look_dir[1])
        grid = observation[2]

        feature_agent_pos = np.zeros((10, 10))
        feature_agent_pos[agent_pos[0], agent_pos[1]] = 1.0
        feature_agent_pos = feature_agent_pos.reshape(10, 10, 1)

        feature_agent_look_pos = np.zeros((10, 10))
        if 0 <= look_pos[0] < 10:
            if 0 <= look_pos[1] < 10:
                feature_agent_look_pos[look_pos[0], look_pos[1]] = 1.0
        feature_agent_look_pos = feature_agent_look_pos.reshape(10, 10, 1)

        grid = grid.reshape(10, 10, -1)

        return np.dstack((grid, feature_agent_look_pos, feature_agent_pos)).reshape(1, 4, 10, 10)

    @staticmethod
    def num_features():
        return 400

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
    def __init__(self, batch_size=1024, max_frames=100000, num_repetitions=1,
                 buffer_size=1000000, eps_start=0.99, eps_end=0.01, eps_frames=100000,
                 learning_rate=1e-3, gamma=0.70, seed=0, episode_len=100):
        super(Agent, self).__init__()

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_repetitions = num_repetitions
        self.gamma = gamma
        self.max_frames = max_frames
        self.seed = 0
        self.episode_len = episode_len

    def train_init(self, env_definition):
        self._env = env_definition.build_env()
        self._eps = self.eps_start
        self._num_frames = 0

        # Set random seeds
        t.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        rand.seed(self.seed)

        # Q-Value approximator and optimizator
        self._behavior_q = QValueLinear(Observation.num_features(), self._env.action_space.n).cuda()
        self._online_q = QValueLinear(Observation.num_features(), self._env.action_space.n).cuda()
        self._optimizer = t.optim.Adam(self._online_q.parameters(), lr=self.learning_rate)
        self._replay = ReplayBuffer(size=self.buffer_size)

        self._env_obs, self._env_rew, self._env_done, _ = self._env.reset()
        self._cur_episode_step = 0
        self._trajectory = []
        self._behavior_q.eval()

    def train_step(self):
        if self.train_is_done():
            return

        if self.train_num_steps() % 1000 == 0:
            self._behavior_q = copy.deepcopy(self._online_q)
            self._behavior_q.eval()

        if self._env_done or self._cur_episode_step >= self.episode_len:
            self._log_writer.add_scalar("train_traj_len", self._cur_episode_step, self.train_num_steps())

            # Compute returns
            q_value = 0.0
            for point in reversed(self._trajectory):
                q_value = point[2] + q_value * self.gamma
                self._replay.add(point[0], point[1], q_value, point[3], point[4])
            self._trajectory = []

            # Train the network (VOLATILE=FALSE IS VERY IMPORTANT)
            if len(self._replay) > 10000:
                obs_batch, act_batch, rew_batch, _, _ = self._replay.sample(self.batch_size)
                obs_batch = t.squeeze(Observation.to_torch(obs_batch)).cuda()
                act_batch = Variable(LongTensor(act_batch), volatile=False).cuda()
                rew_batch = Variable(FloatTensor(rew_batch), volatile=False).cuda()
                for i in range(self.num_repetitions):
                    estimated_values = self._online_q.forward(obs_batch)
                    estimated_values = estimated_values.view(estimated_values.size(0), -1)
                    act_batch = act_batch.view(act_batch.size(0), 1)
                    loss = F.mse_loss(t.gather(estimated_values, dim=-1, index=act_batch), rew_batch)

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    self._log_writer.add_scalar("q_value_loss", loss, self.train_num_steps())

            # Reset the episode
            self._env_obs, self._env_rew, self._env_done, _ = self._env.reset()
            self._cur_episode_step = 0

        self._features = Observation.to_features(self._env_obs, self._env)

        # Select an action in an epsilon-greedy way
        action_values = self._behavior_q.forward(Observation.to_torch(self._features, volatile=True).cuda())
        action = self._env.action_space.sample()
        if np.random.rand() > self._eps:
            action = t.max(action_values, dim=-1)[1].data[0]

        # Take an action and keep current observation
        self._env_obs, self._env_rew, self._env_done, _ = self._env.step(action)
        self._features_after = Observation.to_features(self._env_obs, self._env)

        self._trajectory.append((self._features, action, self._env_rew, self._features_after, self._env_done))

        # Update current observation
        self._features = self._features_after

        # Update epsilon and number of frames
        self._num_frames += 1
        self._cur_episode_step += 1
        self._eps = self.eps_start - (self.eps_start - self.eps_end) * np.clip((self._num_frames / self.eps_frames), 0.0, 1.0)

    def train_num_steps(self):
        return self._num_frames

    def train_is_done(self):
        return self._num_frames >= self.max_frames

    def act(self, observation):
        features = Observation.to_features(observation, self._env)
        action_values = self._behavior_q.forward(Observation.to_torch(features, volatile=True).cuda())
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
        return "ddqn-epis_len_{}-eps_{}-num_reps_{}-lr_{}-gamma_{}-buffer_size_{}-batch_size_{}-max_frames_{}-seed_{}" \
            .format(self.episode_len, self.eps_start, self.num_repetitions, self.learning_rate, self.gamma, self.buffer_size,
                    self.batch_size, self.max_frames, self.seed)