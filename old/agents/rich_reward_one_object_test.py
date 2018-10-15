from env import FindItemsEnv, FindItemsVisualizator
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from utils.training import Session

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


NUM_TESTS = 10000
TEST_SEED_START = 10000
TEST_SEED_END = TEST_SEED_START + NUM_TESTS

model_name = "model_383143.pt-0.pt"
checkpoint = "training/session-agent-1-56-eps-0.95/checkpoints/" + model_name
model = torch.load(checkpoint)

num_successes = 0
num_timesteps = []

for seed in range(TEST_SEED_START, TEST_SEED_END):
	env = FindItemsEnv(10, 10, 3, reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
	env.seed(seed)
	obs, reward, done, _ = env.reset([0])
	timesteps = 0
	while not done and timesteps < 100:
		estimates = model.forward(Observation.to_torch(Observation.to_features(obs, env)))
		action = torch.max(estimates, dim=-1)[1].data[0]
		obs, reward, done, _ = env.step(action)
		timesteps += 1

	num_timesteps.append(timesteps)
	if timesteps != 100:
		num_successes += 1
	if len(num_timesteps) % 100 == 0:
		print("Completed episodes: " + str(len(num_timesteps)))

print("")
print("Success rate: " + str(round(num_successes / NUM_TESTS, 2)) + "%")
print("Average timesteps: " + str(np.mean(num_timesteps)))