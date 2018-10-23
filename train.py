import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

from envs.gridworld.env import FindItemsEnv
from envs.gridworld.instructions import InstructionTokenizer
from envs.builder import build_async_environment
from models.builder import build_policy
from models.utils import to_device, calc_entropy
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from time import time


def take_only_grid(state, num_stack):
    grid = []
    for i in range(num_stack):
        grid.append(state[i][2])
    return np.concatenate(grid, axis=1)
def take_only_grids(states, num_async_envs, num_stack):
    grids = []
    for i in range(num_async_envs):
        grid = take_only_grid(states[i], num_stack)
        grids.append(grid)

    return np.stack(grids)
def take_agent_state(state, num_stack):
    externals = []
    for i in range(num_stack):
        position = np.array(state[i][0]).flatten()
        rotation = np.array(FindItemsEnv.look_to_dir(state[i][1])).flatten()
        externals.append(np.concatenate([position, rotation]))
    return np.concatenate(externals)
def take_agent_states(states, num_async_envs, num_stack):
    externals = []
    for i in range(num_async_envs):
        externals.append(take_agent_state(states[i], num_stack))
    return np.stack(externals)
def compute_returns(bootstrap_value, rewards, masks, gamma=0.99):
    '''
    params:
        bootstrap_value - [batch_size, 1]
        rewards - [num_bootstrap_steps, batch_size]
        masks - [num_bootstrap_steps, batch_size]
    returns:
        returns - [num_bootstrap_steps, batch_size]
    '''
    R = bootstrap_value.squeeze(-1)
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)

    return returns
def sample_instruction(instruction_tokenizer, instructions):
    instruction = random.choice(instructions)
    return instruction_tokenizer.text_to_ids(instruction[0]), instruction[1]
def reset_with_goal(env, goal, at=None):
    if at != None:
        env.send_msg(goal, at)
        return env.reset_at(at)
    else:
        env.send_msg(goal)
        return env.reset()
def benchmark(policy, env):
    policy.eval()

    # Initialize environment
    instruction, goal = sample_instruction(instructions_tokenizer, instructions)
    instruction = Variable(to_device(t.LongTensor(np.tile(instruction, (env.nenvs, 1))), device), volatile=True)
    cur_views, rewards, dones, _ = reset_with_goal(env, goal)
    cur_external_states = Variable(to_device(t.FloatTensor(take_agent_states(cur_views, env.nenvs, args.num_stack)), device), volatile=True)
    cur_views = Variable(to_device(t.FloatTensor(take_only_grids(cur_views, env.nenvs, args.num_stack)), device), volatile=True)
    cur_agent_state = policy.get_initial_state(num_agents)

    num_steps = 0
    trajectory_lengths = [0 for _ in range(env.nenvs)]
    successes = [False for _ in range(env.nenvs)]
    already_dones = [False for _ in range(env.nenvs)]
    masks = []
    while not np.all(already_dones):
        # Act based on current policy
        actions_dist, q_values, cur_agent_state = policy.forward(cur_views, instruction, cur_agent_state, cur_external_states)
        # if np.random.rand() <= eps:
        #     actions = Variable(t.LongTensor(q_values.size()).random_(0, 6), volatile=True)
        # else:
        #     _, actions = t.max(q_values, dim=-1)
        actions = actions_dist.sample()

        cur_views, rewards, dones, _ = env.step(actions.data.cpu().numpy())
        cur_external_states = Variable(to_device(t.FloatTensor(take_agent_states(cur_views, env.nenvs, args.num_stack)), device), volatile=True)
        #print(cur_external_states)
        cur_views = Variable(to_device(t.FloatTensor(take_only_grids(cur_views, env.nenvs, args.num_stack)), device), volatile=True)

        for ind, done in enumerate(dones):
            if done and not already_dones[ind]:
                trajectory_lengths[ind] = num_steps
                successes[ind] = rewards[ind] >= 1.0 # >= 1.0; >= 0.0
                already_dones[ind] = True


        # Save this bootstrap step values
        masks.append(Variable(to_device(t.FloatTensor(1.0-dones), device), volatile=True))
        cur_agent_state = cur_agent_state * masks[-1].unsqueeze(1)

        num_steps += 1

    
    policy.train()
    return successes, trajectory_lengths

# Reproducible research
t.manual_seed(0)
random.seed(0)
np.random.seed(0)


# Command line interface
parser = ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--num-test-envs", type=int, default=4)
parser.add_argument("--num-stack", type=int, default=4)
parser.add_argument("--num-bootstrap", type=int, default=5)
parser.add_argument("--dir-tensorboard", type=str, default=".tensorboard")
args = parser.parse_args()


# training device
device = "cpu"#"gpu" if t.cuda.is_available() else "cpu"


# Training instructions and according environments
instructions = [("Go to the red", [0])]#, ("Go to the green", [1])]
instructions_tokenizer = InstructionTokenizer([instr[0] for instr in instructions], padding_len=6)
reward_scaler = {"name": "min_max", "min_reward": -10.0, "max_reward": 0.0}
train_env = build_async_environment(environment="grid_world", num_async_envs=args.num_envs,
                                    reward_scaler=reward_scaler,
                                    width=10, height=10, num_items=2, instruction=[0],
                                    stack_size=args.num_stack, must_avoid_non_targets=False,
                                    reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS)
test_env = build_async_environment(environment="grid_world", num_async_envs=args.num_test_envs,
                                   reward_scaler=reward_scaler,
                                   width=10, height=10, num_items=2, instruction=[0],
                                   stack_size=args.num_stack, must_avoid_non_targets=False,
                                   reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS)


# Behaviour policy
policy = to_device(build_policy(stack_frames=args.num_stack), device)
policy_optimizer = optim.SGD(policy.parameters(), lr=0.001)


max_frames = 10000000
num_frames = 0

eps_max = 0.95
eps_frames = 1000000
eps = eps_max

num_agents = args.num_envs
num_bootstrap_steps = args.num_bootstrap


# Logging
writer = SummaryWriter(os.path.join(args.dir_tensorboard, "a2c-{}-{}-{}-{}".format(num_agents, num_bootstrap_steps, 2, "every_item")))


# Initialize agent and the 
instruction, goal = sample_instruction(instructions_tokenizer, instructions)
instruction = Variable(to_device(t.LongTensor(np.tile(instruction, (num_agents, 1))), device))
cur_views, rewards, dones, _ = reset_with_goal(train_env, goal)
cur_external_states = Variable(to_device(t.FloatTensor(take_agent_states(cur_views, args.num_envs, args.num_stack)), device))
cur_views = Variable(to_device(t.FloatTensor(take_only_grids(cur_views, args.num_envs, args.num_stack)), device))
cur_agent_state = policy.get_initial_state(num_agents)

# Main loop
while num_frames < max_frames:
    bootstrap_log_probs = []
    bootstrap_q_values = []
    bootstrap_rewards = []
    bootstrap_masks = []
    entropy = 0.0


    # Pretend that agent hidden state is new
    _cur_agent_state = cur_agent_state
    cur_agent_state = policy.get_initial_state(num_agents)
    cur_agent_state.data = cur_agent_state.data


    for _ in range(num_bootstrap_steps):
        # Act based on current policy
        actions_dist, q_values, cur_agent_state = policy.forward(cur_views, instruction, cur_agent_state, cur_external_states)
        # if np.random.rand() <= eps:
        #     actions = Variable(t.LongTensor(q_values.size()).random_(0, 6), volatile=True)#actions_dist.sample()
        # else:
        #     _, actions = t.max(q_values, dim=-1)
        actions = actions_dist.sample()


        cur_views, rewards, dones, _ = train_env.step(actions.data.cpu().numpy())
        cur_external_states = Variable(to_device(t.FloatTensor(take_agent_states(cur_views, args.num_envs, args.num_stack)), device))
        cur_views = Variable(to_device(t.FloatTensor(take_only_grids(cur_views, args.num_envs, args.num_stack)), device))


        # Save this bootstrap step values
        bootstrap_log_probs.append(actions_dist.log_prob(actions))
        bootstrap_q_values.append(q_values)
        bootstrap_rewards.append(Variable(to_device(t.FloatTensor(rewards), device)))
        bootstrap_masks.append(Variable(to_device(t.FloatTensor(1.0-dones), device)))

        entropy += calc_entropy(actions_dist).mean() / num_bootstrap_steps


        cur_agent_state = cur_agent_state * bootstrap_masks[-1].unsqueeze(1)


        num_frames += 1
        eps = eps_max - (eps_max - 0.0) * np.clip((num_frames / eps_frames), 0.0, 1.0)
        if num_frames % 2000 == 0:
            successes, trajectory_lengths = benchmark(policy, test_env)
            success_rate = sum(successes) / len(successes)
            mean_trajectory = np.mean(trajectory_lengths)

            # Write to tensorboard
            writer.add_scalar("success_rate", success_rate, num_frames)
            writer.add_scalar("trajectory_length", mean_trajectory, num_frames)

            # Write to console
            print("Seen frames: {}; Success rate: {}; Trajectory Length: {}".format(num_frames, success_rate, mean_trajectory))
            print("Current entropy: {}; Accumulated for some steps (maybe 1): {}".format(calc_entropy(actions_dist).mean().cpu().data[0], entropy.cpu().data[0]))
            print("Actions distr: {}".format(actions_dist.probs))
            if num_frames % 10000 == 0:
                t.save(policy, os.path.join(".checkpoints", "own_policy_{}.model".format(num_frames)))

        if np.all(dones):
            break

  
    # Predict future returns and compute bootstraps
    _, q_values, _ = policy.forward(cur_views, instruction, cur_agent_state, cur_external_states)
    returns = compute_returns(Variable(t.zeros(q_values.shape), volatile=True), bootstrap_rewards, bootstrap_masks, 0.90)

    returns = t.stack(returns, dim=1).data
    log_probs = t.stack(bootstrap_log_probs, dim=1)
    values = t.stack(bootstrap_q_values, dim=1)

    advantages = returns# - values.data

    actor_loss = (log_probs * Variable(advantages)).mean().neg()
    #critic_loss = F.mse_loss(values, Variable(returns))

    loss = actor_loss#critic_loss#actor_loss #+ critic_loss #0.5 * (critic_loss)#- 0.001 * entropy
    policy_optimizer.zero_grad()
    loss.backward()

    #nn.utils.clip_grad_norm(policy.parameters(), 1.0)
    policy_optimizer.step()

    # Restart done environments
    for ind, done in enumerate(dones):
        if done:
            # Reset the environment with new goals
            new_instruction, new_goal = sample_instruction(instructions_tokenizer, instructions)

            cur_view, reward, done, _ = reset_with_goal(train_env, new_goal, at=ind)
            cur_external_state = Variable(to_device(t.FloatTensor(take_agent_state(cur_view, args.num_stack)), device))
            cur_view = Variable(to_device(t.FloatTensor(take_only_grid(cur_view, args.num_stack)), device))

            cur_views[ind, :] = cur_view
            cur_external_states[ind, :] = cur_external_state
            cur_agent_state[ind, :] = policy.get_initial_state(1).squeeze(0)
    
