"""
This file contains training and testing processes.
"""

import logging
import random
import torch
import gym
import os
import json
import copy
import ctypes

import torch.optim            as optim
import torch.nn               as nn
import torch.nn.functional    as F      # to pay respect
import torch.multiprocessing  as mp
import numpy                  as np

from torch.autograd                                import Variable
from envs.definitions                              import InstructionEnvironmentDefinition
from envs.gridworld_simple.env                     import FindItemsEnvObsOnlyGrid
from envs.gridworld_simple.env                     import FindItemsEnv
from envs.definitions                              import NaturalLanguageInstruction
from envs.definitions                              import Instruction
from envs.gridworld_simple.instructions.tokenizer  import InstructionTokenizer
from envs.goal.env                                 import GoalStatus
from utils.training                                import fix_random_seeds
from utils.training                                import create_experiment_folder
from typing                                        import List, Tuple
from typing                                        import Optional, Dict
from tensorboardX                                  import SummaryWriter
from collections                                   import defaultdict

# Target environment
from experiments.dqn_ga_pe.parameters import env_definition

# Instructions
from experiments.dqn_ga_pe.parameters import instructions
from experiments.dqn_ga_pe.parameters import instructions_level
from experiments.dqn_ga_pe.parameters import tokenizer

# Agent's training and testing parameters
from experiments.dqn_ga_pe.parameters import train_parameters
from experiments.dqn_ga_pe.parameters import test_parameters
from experiments.dqn_ga_pe.parameters import experiment_folder
from experiments.dqn_ga_pe.parameters import TEST_MODE_STOCHASTIC
from experiments.dqn_ga_pe.parameters import TEST_MODE_DETERMINISTIC

# Computational model
from experiments.dqn_ga_pe.model 	   import Model
from experiments.dqn_ga_pe.model      import prepare_model_input
from experiments.dqn_ga_pe.parameters import device

# Training process
class Replay:
	def __init__(self, size=100000):
		self._replay = []
		self._size = size

	def append(self, memento):
		self._replay.append(memento)
		if len(self._replay) >= self._size:
			self._replay = self._replay[-self._size:]

	def sample(self, n):
		if len(self._replay) == 0:
			return []

		if n > len(self._replay):
			return [self._replay[i] for i in np.random.choice(len(self._replay), size=len(self._replay), replace=False)]
		else:
			return [self._replay[i] for i in np.random.choice(len(self._replay), size=n, replace=False)]

def sample_instruction(instructions: List[NaturalLanguageInstruction]) -> NaturalLanguageInstruction:
	ind = np.random.choice(len(instructions))
	return instructions[ind]

def prepare_train_batch(epochs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	size         = len(epochs)
	obs_visual   = [epoch[0][0] for epoch in epochs]
	obs_language = [epoch[0][1] for epoch in epochs]
	obs_step     = [epoch[0][2] for epoch in epochs]
	actions      = [epoch[1] for epoch in epochs]
	rewards		 = [epoch[2] for epoch in epochs]

	result = (
		(torch.tensor(obs_visual).to(device), 
		 torch.tensor(obs_language).squeeze_().to(device),
		 torch.tensor(obs_step).view(size, 1).to(device)
		),
		torch.tensor(actions, dtype=torch.int64).view(size, 1).to(device),
		torch.tensor(rewards).view(size, 1).to(device)
	)

	return result

def train(
	env_definition:      InstructionEnvironmentDefinition,
	instructions:        List[NaturalLanguageInstruction],
	tokenizer:           InstructionTokenizer,

	logdir:              str,
	seed:                int, 
	input_size:          int,

	max_episode_len:     int,
	max_episodes:        int,

	learning_rate:       float,
	gamma:               float,

	eps_start:           float,
	eps_end:             float,
	eps_episodes:        int,

	batch_size:          int,

	replay_size:         int,
	model_switch:        int,
	
	test_every:          int,
	test_repeat:         int):

	fix_random_seeds(seed)

	# Logging
	logger = SummaryWriter(logdir)

	# Model
	model_target  = Model(input_size, max_episode_len)
	model_sampler = Model(input_size, max_episode_len)

	model_target.to(device)
	model_sampler.to(device)

	model_sampler.load_state_dict(model_target.state_dict())
	model_sampler.eval()

	model_target.train()
	optimizer = optim.Adam(model_target.parameters(), lr=learning_rate)

	# Replay
	replay = Replay(replay_size)

	# Training vars
	total_steps     = 0
	total_episodes  = 0
	cur_eps = eps_start

	# Testing vars
	total_successes = []
	total_lengths   = []

	while total_episodes < max_episodes:
		episode_length  = 0
		trajectory      = []

		# Sample new instruction and create an environment
		instruction     = sample_instruction(instructions)
		env             = env_definition.build_env(instruction[1])

		# Reset the environment
		observation, _, done, _ = env.reset()
		observation             = torch.from_numpy(observation).float()
		instruction_idx         = tokenizer.text_to_ids(instruction[0])
		instruction_idx         = np.array(instruction_idx).reshape(1, -1)

		for step in range(max_episode_len):
			episode_length += 1
			total_steps    += 1

			# Forward pass
			with torch.no_grad():
				prev_observation = np.copy(observation)
				model_input 	 = prepare_model_input(prev_observation, instruction_idx, episode_length - 1)
				values 			 = model_sampler(model_input)
			
			# Epsilon-greedy
			if np.random.rand() > cur_eps:
				action = torch.argmax(values, dim=-1).cpu().item()
			else:
				action = env.action_space.sample()

			observation, reward, done,  _ = env.step(action)
			done = done or episode_length >= max_episode_len

			# Collect data
			action_value = values[0, action].cpu().item() # Can be used for td-error further
			trajectory.append(((prev_observation, instruction_idx, episode_length - 1), action, reward, action_value))

			if done:
				break

		total_episodes += 1

		# Calculate reward and move trajectoy to eperience replay
		discounted_reward = 0.0
		for epoch in reversed(trajectory):
			epoch_reward = epoch[2]
			epoch_obs    = epoch[0]
			epoch_action = epoch[1]

			discounted_reward = discounted_reward * gamma + epoch_reward
			replay.append((epoch_obs, epoch_action, discounted_reward))

		# Train target network
		train_obs, train_actions, true_rewards = prepare_train_batch(replay.sample(batch_size))
		train_values = model_target.forward(train_obs)
		train_values = torch.gather(train_values, index=train_actions, dim=-1)
		train_loss   = nn.SmoothL1Loss().forward(train_values, true_rewards)

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()

		# Last reward (should be changed for compound instructions)
		logger.add_scalar("Rollout/Last Reward", trajectory[-1][2], total_steps)
		logger.add_scalar("Rollout/Episode Length", episode_length, total_steps)
		logger.add_scalar("Rollout/Value loss", train_loss.cpu().item(), total_steps)

		print("Seen episodes/frames: {}; {}".format(total_episodes, total_steps))

		# Schedule epsilon
		cur_eps = np.clip(eps_start - (eps_start - eps_end) * (total_episodes / eps_episodes), eps_end, eps_start)

		# Switch models
		if total_episodes % model_switch == 0:
			model_sampler.load_state_dict(model_target.state_dict())
			model_sampler.eval()

		# Testing
		if total_episodes % test_every == 0:
			best_model_traj_len = -np.inf 

			# Test every instruction
			for instruction in instructions:
				instruction_raw   = instruction[0]
				instruction_items = instruction[1]

				instruction_idx = tokenizer.text_to_ids(instruction[0])
				instruction_idx = np.array(instruction_idx).reshape(1, -1)
				
				instruction_successes = []
				instruction_lengths   = []

				env = env_definition.build_env(instruction_items)
				for _ in range(test_repeat):
					num_steps = 0
					observation, reward, done, _ = env.reset()
					while num_steps < max_episode_len and env.goal_status() == GoalStatus.IN_PROGRESS and not done:
						# Forward pass
						with torch.no_grad():
							model_input = prepare_model_input(observation, instruction_idx, num_steps)
							values      = model_sampler(model_input)
						
						# Epsilon-greedy
						#if np.random.rand() > cur_eps:
						action = torch.argmax(values, dim=-1).cpu().item()
						#else:
						#	action = env.action_space.sample()

						observation, reward, done,  _ = env.step(action)
						done = done or episode_length >= max_episode_len

						num_steps += 1

					if env.goal_status() == GoalStatus.SUCCESS:
						instruction_successes.append(1)
						total_successes.append(1)
					elif env.goal_status() == GoalStatus.FAILURE or env.goal_status() == GoalStatus.IN_PROGRESS:
						instruction_successes.append(0)
						total_successes.append(0)
						num_steps = max_episode_len

					instruction_lengths.append(num_steps)
					total_lengths.append(num_steps)

				# Compute means
				mean_traj_len = np.mean(instruction_lengths)
				mean_suc_rate = np.mean(instruction_successes)

				logger.add_scalar("TestTrajectoryLenMean/{}".format(instruction_raw), mean_traj_len, total_episodes)
				logger.add_scalar("TestSuccessRateMean/{}".format(instruction_raw), mean_suc_rate, total_episodes)

			logger.add_scalar("TestTrajectoryLenMean", np.mean(total_lengths), total_episodes)
			logger.add_scalar("TestSuccessRateMean", np.mean(total_successes), total_episodes)

			if np.mean(total_lengths) > best_model_traj_len:
				torch.save(model_sampler.state_dict(), os.path.join(logdir, "best.model"))
				best_model_traj_len = np.mean(total_lengths)

		
args = (
	env_definition,
	instructions,
	tokenizer,

	experiment_folder,
	train_parameters["seed"],
	tokenizer.get_vocabulary_size(),

	train_parameters["max_episode_len"],
	train_parameters["max_episodes"],

	train_parameters["learning_rate"],
	train_parameters["gamma"],

	train_parameters["eps_start"],
	train_parameters["eps_end"],
	train_parameters["eps_episodes"],

	train_parameters["batch_size"],

	train_parameters["replay_size"],
	train_parameters["model_switch"],

	test_parameters["test_every"],
	test_parameters["test_repeat"]
)
train(*args)
