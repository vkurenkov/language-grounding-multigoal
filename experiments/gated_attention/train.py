import logging
import random
import torch
import gym
import os
import json
import copy

import torch.optim            as optim
import torch.nn               as nn
import torch.nn.functional    as F      # to pay respect
import torch.multiprocessing  as mp
import numpy                  as np

from experiments.gated_attention.model             import A3C_LSTM_GA
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



def ensure_shared_grads(model: nn.Module, shared_model: nn.Module):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def sample_instruction(instructions: List[NaturalLanguageInstruction]) -> NaturalLanguageInstruction:
    return instructions[random.randint(0, len(instructions) - 1)]

def train(
    env_definition:      InstructionEnvironmentDefinition,
    shared_model:        A3C_LSTM_GA,
    instructions:        List[NaturalLanguageInstruction],
    tokenizer:           InstructionTokenizer,
    logdir:              str,
    seed:                int, 
    input_size:          int,
    max_episode_len:     int,
    learning_rate:       int,
    gamma:               int,
    tau:                 int,
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
    observation          = torch.from_numpy(observation).float()/255.0
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
                observation, _, _, _ = env.reset()
                observation          = torch.from_numpy(observation).float()/255.0
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
        R = torch.tensor(R)

        # gae = torch.zeros(1, 1)
        rollout_entropy = 0.0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + advantage.pow(2)

            # Generalized Advantage Estimation
            # delta_t = rewards[i] + gamma * \
            #     values[i + 1].data - values[i].data
            # gae = gae * gamma * tau + delta_t

            policy_loss = policy_loss - log_probs[i] * torch.tensor(advantage) - 0.01 * entropies[i]
            rollout_entropy += entropies[i].data[0]

        rollout_entropy /= len(rewards)
        value_loss      /= len(rewards)

        optimizer.zero_grad()

        #logger.add_scalar("Rollout/GAE Reward", gae.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Mean Entropy", rollout_entropy, total_steps)
        logger.add_scalar("Rollout/Policy loss", policy_loss.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Value loss", value_loss.data[0, 0], total_steps)
        logger.add_scalar("Rollout/Total loss", policy_loss.data[0, 0] + 0.5 * value_loss.data[0, 0], total_steps)

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        if done:
            print("Agent #{}. Seen episodes/frames: {}; {}".format(logdir[-1], total_episodes, total_steps))
            logger.add_scalar("Episode/Reward (sum)", np.sum(episode_rewards), total_episodes)
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
    seed:            int):
    
    # Allow training agents to work
    is_testing_iteration_running.set()

    # Test must be reproducible
    fix_random_seeds(seed)

    logger = SummaryWriter(logdir)

    model               = copy.deepcopy(shared_model)
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

            # Synchronize models
            model.load_state_dict(shared_model.state_dict())

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
                            observation     = torch.from_numpy(observation).float()/255.0
                            tx = torch.tensor(np.array([num_steps]), dtype=torch.int64)
                            _, logit, (hx, cx) = model((
                                                    torch.tensor(observation).unsqueeze(0),
                                                    torch.tensor(instruction_idx),
                                                    (tx, hx, cx)
                                                ))
                            prob   = F.softmax(logit, dim=-1)
                            action = prob.multinomial(1).data
                            action = action.numpy()[0, 0]

                        observation, reward, done, _ = env.step(action)
                        num_steps += 1

                    if env.goal_status() == GoalStatus.SUCCESS:
                        instruction_successes.append(1)
                        total_successes.append(1)
                    elif env.goal_status() == GoalStatus.FAILURE:
                        instruction_successes.append(0)
                        total_successes.append(0)
                        num_steps = max_episode_len

                    instruction_lengths.append(num_steps)
                    total_lengths.append(num_steps)

                logger.add_scalar("TestTrajectoryLenMean/{}".format(instruction_raw), np.mean(instruction_lengths), waited_episodes)
                logger.add_scalar("TestSuccessRateMean/{}".format(instruction_raw), np.mean(instruction_successes), waited_episodes)

            logger.add_scalar("TestTrajectoryLenMean", np.mean(total_lengths), waited_episodes)
            logger.add_scalar("TestSuccessRateMean", np.mean(total_successes), waited_episodes)
                
            if np.mean(total_lengths) > best_model_traj_len:
                torch.save(model, os.path.join(logdir, "best.model"))

            is_testing_iteration_running.set()


def get_level1_instructions() -> List[NaturalLanguageInstruction]:
    dataset_path = os.path.join(get_this_file_path(), "instructions", "level1.json")
    with open(dataset_path, mode="r") as f:
        dataset = json.load(f)

    return [(instruction["raw"], instruction["objects_real_order"]) for instruction in dataset["instructions"]]

def get_instructions_tokenizer(instructions: NaturalLanguageInstruction) -> InstructionTokenizer:
    return InstructionTokenizer([instr[0] for instr in instructions])


def get_this_file_path() -> str:
    return os.path.dirname(__file__)

def unroll_parameters_in_str(parameters: Dict) -> str:
    unrolled = ""
    for key in sorted(parameters.keys()):
        unrolled += "{}_{}-".format(key, parameters[key])

    return unrolled[0:-1] # I know, I know. Not the most elegant way.


# Target environment
env_definition = InstructionEnvironmentDefinition(
                        FindItemsEnvObsOnlyGrid,
                        width=10, height=10, num_items=3,
                        must_avoid_non_targets=True,
                        reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM,
                        fixed_positions=[(0, 0,), (5, 5), (3, 3), (7, 7)]
)

# Target instructions
instructions = get_level1_instructions()
tokenizer    = get_instructions_tokenizer(instructions)

# Agent set-up
agent_parameters = {
    "max_episodes":        150000,
    "max_episode_len":     50,
    "num_processes":       4,
    "learning_rate":       0.001,
    "gamma":               0.95,
    "tau":                 0.00,
    "num_bootstrap_steps": 50,
    "seed":                0
}
fix_random_seeds(agent_parameters["seed"])

# Agent model
agent = A3C_LSTM_GA(tokenizer.get_vocabulary_size(), agent_parameters["max_episode_len"])
agent.share_memory()

# Logging set-up
experiment_folder = create_experiment_folder(
                        os.path.join(get_this_file_path(), "logs"), 
                        env_definition.name(), 
                        unroll_parameters_in_str(agent_parameters)
)

# Testing
episodes_completion          = mp.Queue()
is_testing_iteration_running = mp.Event()

testing_parameters = {
    "test_every":  100,
    "test_repeat": 5,
    "seed":        1337
}
test_args = (
    env_definition,
    instructions,
    tokenizer,
    agent,
    experiment_folder,
    testing_parameters["test_every"],
    testing_parameters["test_repeat"],
    agent_parameters["max_episodes"] * agent_parameters["num_processes"],
    agent_parameters["max_episode_len"],
    testing_parameters["seed"]
)
test_process        = mp.Process(target=test, args=test_args)
test_process.start()

# Training
training_processes = []
for rank in range(1, agent_parameters["num_processes"] + 1):
    args = (
        env_definition,
        agent,
        instructions,
        tokenizer,
        os.path.join(experiment_folder, "agent_{}".format(rank)),
        agent_parameters["seed"] + rank,
        tokenizer.get_vocabulary_size(),
        agent_parameters["max_episode_len"],
        agent_parameters["learning_rate"],
        agent_parameters["gamma"],
        agent_parameters["tau"],
        agent_parameters["num_bootstrap_steps"],
        agent_parameters["max_episodes"]
    )

    p = mp.Process(target=train, args=args)
    p.start()
    training_processes.append(p)

for p in training_processes:
    p.join()
test_process.join()