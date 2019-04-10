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
from collections                                   import defaultdict, deque

# Target environment
from experiments.dueling_dqn_ga.parameters import env_definition

# Instructions
from experiments.dueling_dqn_ga.parameters import instructions
from experiments.dueling_dqn_ga.parameters import instructions_level
from experiments.dueling_dqn_ga.parameters import tokenizer

# Agent's training and testing parameters
from experiments.dueling_dqn_ga.parameters import train_parameters
from experiments.dueling_dqn_ga.parameters import test_parameters
from experiments.dueling_dqn_ga.parameters import experiment_folder
from experiments.dueling_dqn_ga.parameters import TEST_MODE_STOCHASTIC
from experiments.dueling_dqn_ga.parameters import TEST_MODE_DETERMINISTIC

# Computational model
from experiments.dueling_dqn_ga.model 	 import Model
from experiments.dueling_dqn_ga.model      import prepare_model_input
from experiments.dueling_dqn_ga.parameters import device

# Training process
class Replay:
	def __init__(self, size=100000):
		self._replay   = [None] * size
		self._size     = size

		self._cur_size = 0
		self._index    = 0

	def append(self, memento):
		self._replay[self._index] = memento
		self._index    = (self._index + 1) % self._size
		self._cur_size = min(self._cur_size + 1, self._size)
		
	def sample(self, n):
		if self._cur_size == 0:
			return []

		idxs = np.random.choice(self._cur_size, n).tolist()
		return [self._replay[i] for i in idxs]

# PER
# Several priorities possible:
#	1 - Based on the prediction error (TD-error?)
#   2 - Based on the episodic return  (the more we get the more focus)
# Several approaches possible:
#	1 - Proportion-based
#	2 - Ranking-based
class PrioritizedReplay:
	pass

def sample_instruction(instructions: List[NaturalLanguageInstruction]) -> NaturalLanguageInstruction:
	ind = np.random.choice(len(instructions))
	return instructions[ind]

def prepare_train_batch(epochs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	size         = len(epochs)
	
	# Collect stacked frames (logic is a bit messy)
	# 1 - move stacked observations to the pytorch device
	obs_visual = [[torch.tensor(obs).to(device).unsqueeze(0) for obs in epoch[0][0]] for epoch in epochs]
	# 2 - put frames at the same time into one batch
	stacked_frames = len(obs_visual[0])
	visuals = []
	for i in range(stacked_frames):
		frames = [obs[i] for obs in obs_visual]
		frames = torch.cat(frames, dim=0)
		visuals.append(frames)

	obs_language = [epoch[0][1] for epoch in epochs]
	actions      = [epoch[1] for epoch in epochs]
	rewards		 = [epoch[2] for epoch in epochs]

	#print(obs_visual[0])
	result = (
		(visuals, torch.tensor(obs_language).squeeze_().to(device)),
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
	test_repeat:         int,
	
	stack_frames:        int):

	fix_random_seeds(seed)

	# Logging
	logger = SummaryWriter(logdir)

	# Model
	model_target  = Model(input_size, stack_frames, max_episode_len)
	model_sampler = Model(input_size, stack_frames, max_episode_len)

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
	best_model_traj_len = np.inf 

	while total_episodes < max_episodes:
		episode_lengths = []
		episode_rewards = []
		for instruction in instructions:
			episode_length  = 0
			trajectory      = []

			# Sample new instruction and create an environment
			#instruction     = sample_instruction(instructions)
			env             = env_definition.build_env(instruction[1])

			# Reset the environment
			observation, _, done, _ = env.reset()
			observation             = torch.from_numpy(observation).float()
			instruction_idx         = tokenizer.text_to_ids(instruction[0])
			instruction_idx         = np.array(instruction_idx).reshape(1, -1)

			# Hindsight instructions
			hindsight_instructions = defaultdict(list)

			# Keep last frames
			last_frames = deque([observation for _ in range(stack_frames)], stack_frames)

			for step in range(max_episode_len):
				episode_length += 1
				total_steps    += 1

				# Forward pass
				with torch.no_grad():
					prev_observation = list(last_frames)#np.copy(observation)
					model_input 	 = prepare_model_input(prev_observation, instruction_idx)
					values 			 = model_sampler(model_input)
				
				# Epsilon-greedy
				if np.random.rand() > cur_eps:
					action = torch.argmax(values, dim=-1).cpu().item()
				else:
					action = env.action_space.sample()

				observation, reward, done,  _ = env.step(action)
				done = done or episode_length >= max_episode_len

				# Check if it was the last step and we could not successfuly eecute the instruction
				if episode_length >= max_episode_len and done and reward < 10:
					reward += -10

				# Collect data
				last_frames.append(observation)
				action_value = values[0, action].cpu().item() # Can be used for td-error further
				trajectory.append(((prev_observation, instruction_idx), action, reward, action_value))

				# Iterate over all instructions
				# for instruction in instructions:
				# 	items = instruction[1]
				# 	instr = instruction[0]
				# 	# If this instruction is not failed
				# 	# We can use the reward function
				# 	# (Makes sense in dense reward setting, otherwise you should also check for subgoal completion in the end)
				# 	if not env.is_instruction_over(items):
				# 		hindsight_instructions[instr].append(env.reward_per_instruction(items))

				if done:
					break

			total_episodes += 1

			# Calculate reward and move online trajectoy to eperience replay
			discounted_reward_episode = 0.0
			for epoch in reversed(trajectory):
				epoch_reward = epoch[2]
				epoch_obs    = epoch[0]
				epoch_action = epoch[1]

				discounted_reward_episode = discounted_reward_episode * gamma + epoch_reward
				replay.append((epoch_obs, epoch_action, discounted_reward_episode))
			
			# Save stats for current iteration
			episode_rewards.append(discounted_reward_episode)
			episode_lengths.append(episode_length)

			# Calculate reward and move hindsihgt trajectories to eperiene replay
			# for instruction in instructions:
			# 	instr 	    = instruction[0]
			# 	instruction_idx         = tokenizer.text_to_ids(instr)
			# 	instruction_idx         = np.array(instruction_idx).reshape(1, -1)

			# 	rewards    = reversed(hindsight_instructions[instr])
			# 	discounted_reward = 0.0
			# 	for ind, epoch_reward in enumerate(rewards):
			# 		epoch        = trajectory[-(ind + 1)]
			# 		epoch_obs    = (epoch[0][0], instruction_idx) # Take grid observation but put another instruction
			# 		epoch_action = epoch[1]

			# 		discounted_reward = discounted_reward * gamma + epoch_reward
			# 		replay.append((epoch_obs, epoch_action, discounted_reward))

		# Train target network
		train_obs, train_actions, true_rewards = prepare_train_batch(replay.sample(batch_size))
		train_values = model_target.forward(train_obs)
		train_values = torch.gather(train_values, index=train_actions, dim=-1)
		train_loss   = nn.MSELoss().forward(train_values, true_rewards)

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()

		# Discounted reward for the episode
		logger.add_scalar("Rollout/Last Reward", np.mean(episode_rewards), total_steps)
		logger.add_scalar("Rollout/Episode Length", np.mean(episode_lengths), total_steps)
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
					last_frames = deque([observation] * stack_frames, stack_frames)
					while num_steps < max_episode_len and env.goal_status() == GoalStatus.IN_PROGRESS and not done:
						# Forward pass
						with torch.no_grad():
							model_input = prepare_model_input(list(last_frames), instruction_idx)
							values      = model_sampler(model_input)
						
						# Epsilon-greedy
						if np.random.rand() > cur_eps:
							action = torch.argmax(values, dim=-1).cpu().item()
						else:
							action = env.action_space.sample()

						observation, reward, done,  _ = env.step(action)
						done = done or episode_length >= max_episode_len

						num_steps += 1
						last_frames.append(observation)

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

			if np.mean(total_lengths) < best_model_traj_len:
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
	test_parameters["test_repeat"],

	train_parameters["stack_frames"]
)
train(*args)