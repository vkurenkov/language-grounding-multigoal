import os
import sys
import pickle

from tensorboardX import SummaryWriter
from envs.gridworld.env import FindItemsEnv
from agents.dqn import DQNEpsilonGreedyAgent
from benchmarks.benchmark import TrajectoryLengthBenchmark
from argparse import ArgumentParser


class EnvironmentDefinition:
    def __init__(self, env_constructor, **kwargs):
        self._env_constructor = env_constructor
        self._kwargs = kwargs

    def build_env(self):
        return self._env_constructor(**self._kwargs)


def save_agent(agent, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="wb") as f:
        pickle.dump(agent, f)


# Command line interface
parser = ArgumentParser()
parser.add_argument("--dir-tensorboard", type=str, default=".tensorboard")
parser.add_argument("--dir-checkpoints", type=str, default=".checkpoints")
parser.add_argument("--benchmark-trials", type=int, default=10)
parser.add_argument("--benchmark-every", type=int, default=10000)
parser.add_argument("--checkpoint-every", type=int, default=10000)
args = parser.parse_args()

# Define training process
env_definition = EnvironmentDefinition(FindItemsEnv, width=10, height=10, num_items=2,
                                       instruction=[0], must_avoid_non_targets=False,
                                       reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS)
agent = DQNEpsilonGreedyAgent()
writer = SummaryWriter(os.path.join(args.dir_tensorboard, agent.name()))

# Define benchmarks
trajectory_length_benchmark = TrajectoryLengthBenchmark(env_definition, n_trials=args.benchmark_trials)

agent.train_init(env_definition)
while not agent.train_is_done():
    agent.train_step()

    # Benchmarking
    if agent.train_num_steps() % args.benchmark_every == 0:
        trajectory_length = trajectory_length_benchmark(agent)

        writer.add_scalar("trajectory_length", trajectory_length, agent.train_num_steps())
        print("#{}, Mean trajectory length: {}".format(agent.train_num_steps(), trajectory_length))

    # Checkpointing
    if agent.train_num_steps() % args.checkpoint_every == 0:
        save_agent(agent, os.path.join(args.dir_checkpoints, agent.name() + "-{}.agent".format(agent.train_num_steps()))) 