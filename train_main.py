import os
import sys
import pickle

from tensorboardX import SummaryWriter
from envs.gridworld_simple.env import FindItemsEnv
from envs.definitions import GoalEnvironmentDefinition
from agents.dqn import DQNEpsilonGreedyAgent
from agents.perfect import PerfectAgent
from benchmarks.benchmark import SuccessTrajectoryLengthBenchmark, SuccessRateBenchmark
from argparse import ArgumentParser


AGENT_PERFECT = "perfect"
AGENT_DQN     = "dqn"


def save_agent(agent, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="wb") as f:
        pickle.dump(agent, f)


# Command line interface
parser = ArgumentParser()
parser.add_argument("--agent", type=str, default=AGENT_PERFECT)
parser.add_argument("--dir-tensorboard", type=str, default=".tensorboard")
parser.add_argument("--dir-checkpoints", type=str, default=".checkpoints")
parser.add_argument("--benchmark-trials", type=int, default=10)
parser.add_argument("--benchmark-every", type=int, default=10000)
parser.add_argument("--checkpoint-every", type=int, default=10000)
args = parser.parse_args()

# Define training process
env_definition = GoalEnvironmentDefinition(FindItemsEnv, width=10, height=10, num_items=2,
                                       instruction=[1], must_avoid_non_targets=True,
                                       reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
                                       #fixed_positions=[(0,0), (5, 5), (3, 3)],
                                       #fixed_look="EAST")

# Define an agent
if args.agent == AGENT_DQN:
    agent = DQNEpsilonGreedyAgent(max_frames=50000000,eps_frames=25000000,gamma=0.90,batch_size=2048,learning_rate=1e-3)
else:
    agent = PerfectAgent()

writer = SummaryWriter(os.path.join(args.dir_tensorboard, env_definition.name(), agent.name()))

# Define benchmarks
trajectory_length_benchmark = SuccessTrajectoryLengthBenchmark(
    env_definition, n_trials=args.benchmark_trials
)
success_rate_benchmark      = SuccessRateBenchmark(
    env_definition, n_trials=args.benchmark_trials
)

agent.log_init(writer)
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

# Final benchmarking
print("Mean benchmark trajectory length: {}".format(trajectory_length_benchmark(agent)))
print("Mean success rate: {}".format(success_rate_benchmark(agent)))
