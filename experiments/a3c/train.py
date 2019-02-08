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
from experiments.a3c.parameters import env_definition

# Instructions
from experiments.a3c.parameters import instructions
from experiments.a3c.parameters import instructions_level
from experiments.a3c.parameters import tokenizer

# Agent's training and testing parameters
from experiments.a3c.parameters import train_parameters
from experiments.a3c.parameters import test_parameters
from experiments.a3c.parameters import experiment_folder
from experiments.a3c.parameters import TEST_MODE_STOCHASTIC
from experiments.a3c.parameters import TEST_MODE_DETERMINISTIC

# Computational model
from experiments.a3c.model import A3C_LSTM_GA

# Training process
def ensure_shared_grads(model: nn.Module, shared_model: nn.Module):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def sample_instruction(instructions: List[NaturalLanguageInstruction]) -> NaturalLanguageInstruction:
    ind = np.random.choice(len(instructions))
    return instructions[ind]

def train(
    env_definition:      InstructionEnvironmentDefinition,
    shared_model:        A3C_LSTM_GA,
    instructions:        List[NaturalLanguageInstruction],
    tokenizer:           InstructionTokenizer,
    logdir:              str,
    seed:                int, 
    input_size:          int,
    max_episode_len:     int,
    learning_rate:       float,
    gamma:               float,
    tau:                 float,
    entropy_coeff:       float,
    num_bootstrap_steps: int,
    max_episodes:        int):

    # Agents should start from different seeds
    # Otherwise it will lead to the same experience (not desirable at all)
    fix_random_seeds(seed)

    logger = SummaryWriter(logdir)

    model = A3C_LSTM_GA(input_size, max_episode_len)
    model.train()
    optimizer = optim.SGD(shared_model.parameters(), lr=learning_rate)


    instruction     = sample_instruction(instructions)
    env             = env_definition.build_env(instruction[1])


    observation, _, _, _ = env.reset()
    observation          = torch.from_numpy(observation).float()
    instruction_idx      = tokenizer.text_to_ids(instruction[0])
    instruction_idx      = np.array(instruction_idx)
    instruction_idx      = torch.from_numpy(instruction_idx).view(1, -1)


    done            = True
    episode_length  = 0
    episode_rewards = []
    total_steps     = 0
    total_episodes  = 0

    while total_episodes < max_episodes:
        # Holt while testing is performed
        is_testing_iteration_running.wait()

        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:
            episode_length  = 0
            episode_rewards = []
            cx = torch.zeros(1, 256, requires_grad=True)
            hx = torch.zeros(1, 256, requires_grad=True)

            episodes_completion.put(1)
        else:
            cx = cx.clone().detach().requires_grad_(True)
            hx = hx.clone().detach().requires_grad_(True)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(num_bootstrap_steps):
            episode_length += 1
            total_steps      += 1

            tx = torch.tensor(np.array([episode_length]), dtype=torch.int64)
            
            value, logit, (hx, cx) = model((torch.tensor(observation).unsqueeze(0),
                                            torch.tensor(instruction_idx),
                                            (tx, hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial(1).data
            log_prob = log_prob.gather(1, torch.tensor(action))

            action = action.numpy()[0, 0]
            observation, reward, done,  _ = env.step(action)

            done = done or episode_length >= max_episode_len

            if done:
                instruction     = sample_instruction(instructions)
                env             = env_definition.build_env(instruction[1])

                observation, _, _, _ = env.reset()
                observation          = torch.from_numpy(observation).float()
                instruction_idx      = tokenizer.text_to_ids(instruction[0])
                instruction_idx      = np.array(instruction_idx)
                instruction_idx      = torch.from_numpy(instruction_idx).view(1, -1)

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            episode_rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            tx = torch.tensor(np.array([episode_length]), dtype=torch.int64)
            value, _, _ = model((torch.from_numpy(observation).unsqueeze(0),
                                 instruction_idx.clone().detach(), (tx, hx, cx)))
            R = value.data

        values.append(torch.tensor(R))
        policy_loss = 0
        value_loss = 0
        R = torch.tensor(rewards[-1])

        gae = torch.zeros(1, 1)
        rollout_entropy = 0.0
        for i in reversed(range(len(rewards))):
            R = gamma * R
            advantage = R - values[i]
            value_loss = value_loss + advantage.pow(2)

            policy_loss = policy_loss - \
                log_probs[i] * torch.tensor(advantage) - entropy_coeff * entropies[i]
            rollout_entropy += entropies[i].data[0]

        rollout_entropy /= len(rewards)
        value_loss      /= len(rewards)

        optimizer.zero_grad()

        logger.add_scalar("Rollout/Mean Entropy", rollout_entropy, total_steps)
        logger.add_scalar("Rollout/Policy loss", policy_loss.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Value loss", value_loss.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Total loss", policy_loss.data[0, 0] + 0.5 * value_loss.data[0, 0], total_steps)

        (policy_loss + 0.5 * value_loss).backward()

        ensure_shared_grads(model, shared_model)

        # Log training parameters
        if rank % 8 == 0:
            logger.add_histogram("Grads/Policy-LSTM-Input",  shared_model.lstm.weight_ih.grad, total_steps)
            logger.add_histogram("Grads/Policy-LSTM-Hidden",  shared_model.lstm.weight_hh.grad, total_steps)
            logger.add_histogram("Grads/Instruction-GRU-Input",  shared_model.gru.weight_ih_l0.grad, total_steps)
            logger.add_histogram("Grads/Instruction-GRU-Hidden", shared_model.gru.weight_hh_l0.grad, total_steps)

        optimizer.step()

        if done:
            print("Agent #{}. Seen episodes/frames: {}; {}".format(logdir[-1], total_episodes, total_steps))
            logger.add_scalar("Episode/Reward", R.item(), total_episodes)
            logger.add_scalar("Episode/Length", episode_length, total_episodes)
            total_episodes += 1

def test(
    env_definition:  InstructionEnvironmentDefinition,
    instructions:    List[NaturalLanguageInstruction],
    tokenizer:       InstructionTokenizer,
    shared_model:    A3C_LSTM_GA,
    logdir:          str,
    test_every:      int,
    test_repeat:     int,
    max_episodes:    int,
    max_episode_len: int,
    mode:            str,
    seed:            int):
    
    # Allow training agents to work
    is_testing_iteration_running.set()

    # Test must be reproducible
    fix_random_seeds(seed)

    logger = SummaryWriter(logdir)

    best_model_traj_len = -np.inf 

    waited_episodes = 0

    while waited_episodes <= max_episodes:
        # Wait until training agents completed one or more episodes
        waited_episodes += episodes_completion.get()

        total_successes = []
        total_lengths   = []
        if waited_episodes % test_every == 0:
            # Training agents must wait until the testing is over
            is_testing_iteration_running.clear()

            print("Testing iteration {}".format(waited_episodes))

            # Test every instruction
            for instruction in instructions:
                instruction_raw   = instruction[0]
                instruction_items = instruction[1]

                instruction_idx = tokenizer.text_to_ids(instruction[0])
                instruction_idx = np.array(instruction_idx)
                instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)
                

                instruction_successes = []
                instruction_lengths   = []

                env = env_definition.build_env(instruction_items)
                for _ in range(test_repeat):
                    num_steps = 0
                    cx        = torch.zeros(1, 256)
                    hx        = torch.zeros(1, 256)

                    observation, reward, done, _ = env.reset()
                    while num_steps < max_episode_len and env.goal_status() == GoalStatus.IN_PROGRESS and not done:
                        with torch.no_grad():
                            observation     = torch.from_numpy(observation).float()
                            tx = torch.tensor(np.array([num_steps + 1]), dtype=torch.int64)
                            _, logit, (hx, cx) = shared_model((
                                                    torch.tensor(observation).unsqueeze(0),
                                                    torch.tensor(instruction_idx),
                                                    (tx, hx, cx)
                                                ))
                            prob   = F.softmax(logit, dim=-1)

                            if mode == TEST_MODE_DETERMINISTIC:
                                action = torch.argmax(prob, dim=-1).item()
                            elif mode == TEST_MODE_STOCHASTIC:
                                action = prob.multinomial(1).data.numpy()[0, 0]

                        observation, reward, done, _ = env.step(action)
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

                logger.add_scalar("{}-TestTrajectoryLenMean/{}".format(mode, instruction_raw), mean_traj_len, waited_episodes)
                logger.add_scalar("{}-TestSuccessRateMean/{}".format(mode, instruction_raw), mean_suc_rate, waited_episodes)

            logger.add_scalar("{}-TestTrajectoryLenMean".format(mode), np.mean(total_lengths), waited_episodes)
            logger.add_scalar("{}-TestSuccessRateMean".format(mode), np.mean(total_successes), waited_episodes)

            if np.mean(total_lengths) > best_model_traj_len:
                torch.save(shared_model.state_dict(), os.path.join(logdir, "best.model"))
                best_model_traj_len = np.mean(total_lengths)

            is_testing_iteration_running.set()

fix_random_seeds(train_parameters["seed"])

agent = A3C_LSTM_GA(tokenizer.get_vocabulary_size(), train_parameters["max_episode_len"])
agent.share_memory()

episodes_completion          = mp.Queue()
is_testing_iteration_running = mp.Event()
test_args = (
    env_definition,
    instructions,
    tokenizer,
    agent,
    experiment_folder,
    test_parameters["test_every"],
    test_parameters["test_repeat"],
    train_parameters["max_episodes"] * train_parameters["num_processes"],
    train_parameters["max_episode_len"],
    test_parameters["mode"],
    test_parameters["seed"]
)
test_process = mp.Process(target=test, args=test_args)
test_process.start()

training_processes = []
for rank in range(1, train_parameters["num_processes"] + 1):
    args = (
        env_definition,
        agent,
        instructions,
        tokenizer,
        os.path.join(experiment_folder, "agent_{}".format(rank)),
        train_parameters["seed"] + rank,
        tokenizer.get_vocabulary_size(),
        train_parameters["max_episode_len"],
        train_parameters["learning_rate"],
        train_parameters["gamma"],
        train_parameters["tau"],
        train_parameters["entropy_coeff"],
        train_parameters["num_bootstrap_steps"],
        train_parameters["max_episodes"]
    )

    p = mp.Process(target=train, args=args)
    p.start()
    training_processes.append(p)
for p in training_processes:
    p.join()
test_process.join()
