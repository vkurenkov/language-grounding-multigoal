from env import FindItemsEnv
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from utils.training import Session
from copy import deepcopy

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as np_rand


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
		return 304

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
				return [self._replay[i] for i in np_rand.choice(len(self._replay), size=len(self._replay), replace=False)]
			else:
				return [self._replay[i] for i in np_rand.choice(len(self._replay), size=n, replace=False)]
		else:
			probs = np.array([abs(memento[2]) for memento in self._replay])
			probs = probs / np.sum(probs)
			if n > len(self._replay):
				return [self._replay[i] for i in np_rand.choice(len(self._replay), size=len(self._replay), replace=False, p=probs)]
			else:
				return [self._replay[i] for i in np_rand.choice(len(self._replay), size=n, replace=False, p=probs)]


# Constants
NUM_ACTIONS = 6

EPSILON_START = 0.95
EPSILON_FINISH = 0.01
FRAMES = 400000
DISCOUNT = 0.98

REPORT_EPISODES = 25
CHECKPOINT_FRAMES = 50000
EVALUATE_EPISODES = 100

NUM_TRAIN_REPETITIONS = 2

epsilons = reversed([0.95])
seeds = range(0, 10000)
for eps in epsilons:
	seed = -1
	with Session("agent-1-56-eps-" + str(eps)).start() as sess:
		sess.switch_group()

		EPSILON_START = eps
		
		# Log
		sess.log("Epsilon" + str(eps) + " started.\n")
		sess.log_flush()

		# Networks
		behaviour_q = QValue(Observation.num_features(), NUM_ACTIONS)
		target_q = QValue(Observation.num_features(), NUM_ACTIONS)
		optimizer = torch.optim.Adam(behaviour_q.parameters())
		replay = Replay(unusual_sampling=False)

		# Environment
		env = FindItemsEnv(10, 10, 3, reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS)
		num_frames = 0
		last_report = 0
		last_checkpoint = 0
		epsilon = EPSILON_START
		timesteps = []
		losses = []

		while num_frames < FRAMES:

			# Set simulation seed
			seed += 1
			env.seed(seed % len(seeds))

			# Retrieve first observation
			obs, reward, done, _ = env.reset([0])
			features = Observation.to_features(obs, env)

			# Now init agent seeds
			np_rand.seed(seed)
			torch.manual_seed(seed)

			# Track current trajectory
			trajectory = []
			num_timesteps = 0

			while not done and num_timesteps <= 1000:
				# Select an action in an epsilon-greedy way
				action_values = behaviour_q.forward(Observation.to_torch(features, volatile=True))
				action = np_rand.random_integers(0, NUM_ACTIONS-1)
				if np_rand.rand() > epsilon:
					action = torch.max(action_values, dim=-1)[1].data[0]

				# Take an action and keep current observation
				obs, reward, done, _ = env.step(action)
				features_after = Observation.to_features(obs, env)
				trajectory.append((features, action, reward, features_after))

				# Update current observation
				features = features_after

				# Update epsilon and number of frames
				num_frames += 1
				num_timesteps += 1
				epsilon = EPSILON_START - (EPSILON_START - EPSILON_FINISH) * (num_frames / FRAMES)

			# Monte-carlc return bleat
			value = 0.0
			for i, point in enumerate(reversed(trajectory)):
				value = point[2]
				replay.append((point[0], point[1], value))

			# Train the network
			for i in range(NUM_TRAIN_REPETITIONS):
				point_features = None
				point_actions = None
				point_returns = None
				samples = replay.sample(128)
				for sample in samples:
					if point_features is None:
						point_features = Observation.to_torch(sample[0])
						point_actions = Variable(LongTensor([[int(sample[1])]]))
						point_returns = Variable(FloatTensor([[sample[2]]]))
					else:
						point_features = torch.cat((point_features, Observation.to_torch(sample[0])), 0)
						point_actions = torch.cat((point_actions, Variable(LongTensor([[int(sample[1])]]))), 0)
						point_returns = torch.cat((point_returns, Variable(FloatTensor([[sample[2]]]))), 0)

				if len(samples) > 0:
					estimated_values = behaviour_q.forward(point_features)
					loss = torch.mean((point_returns - torch.gather(estimated_values, dim=-1, index=point_actions)) ** 2)
					losses.append(loss.data[0])
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

			sess.timesteps(num_timesteps)
			sess.reward(num_frames) # No hidden meaning here, just to retrieve further
			timesteps = sess.get_timesteps()

			# Report current progress
			if len(timesteps) % REPORT_EPISODES == 0:
				averaged = np.mean(timesteps[-REPORT_EPISODES:])

				sess.log("Current progress: " + str(round((num_frames / FRAMES) * 100, 2)) + "%")
				sess.log("\n")
				sess.log("Current epsilon: " + str(epsilon))
				sess.log("\n")
				sess.log("Last 50 episodes timesteps averaged: " + str(averaged))
				sess.log("\n")
				sess.log("Last 50 episodes loss averaged: " + str(np.mean(losses)))
				sess.log("\n")
				sess.log("Frames seen: " + str(num_frames))
				sess.log("\n")

				if averaged <= 10 and epsilon <= 0.05:
					sess.log("Success!!!")
					sess.log("\n")
					sess.checkpoint_model(behaviour_q, "model_" + str(num_frames) + ".pt")
					sess.log("Saved the model: " + str(num_frames))
					sess.log("\n")
					break

				last_report = last_report
				losses = []
				sess.log_flush()


			# Save current network
			if num_frames - last_checkpoint >= CHECKPOINT_FRAMES:
				sess.checkpoint_model(behaviour_q, "model_" + str(num_frames) + ".pt")
				sess.log("Saved the model: " + str(num_frames))
				sess.log("\n")
				sess.log_flush()
				last_checkpoint = num_frames