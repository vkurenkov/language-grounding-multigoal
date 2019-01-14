import os
import sys
import pickle

from argparse                  import ArgumentParser
from tensorboardX              import SummaryWriter
from envs.gridworld_simple.env import FindItemsEnv, FindItemsVisualizator
from envs.gridworld_simple.env import FindItemsEnvObsOnlyGrid
from envs.definitions          import InstructionEnvironmentDefinition
from agents.perfect            import PerfectAgent
from agents.chaplot.agent      import GatedAttentionAgent
from benchmarks.benchmark      import SuccessTrajectoryLengthBenchmark
from benchmarks.benchmark      import SuccessRateBenchmark
from utils.training            import fix_random_seeds

from envs.gridworld_simple.instructions.tokenizer import InstructionTokenizer


import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


AGENT_PERFECT          = "perfect"
AGENT_GATED_ATTENTTION = "a3c_lstm_gated_attention" # Chaplot et al.
AGENT_A3C_LSTM         = "a3c_lstm"                 # Misra et al.


def save_agent(agent, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="wb") as f:
        pickle.dump(agent, f)


# Command line interface
parser = ArgumentParser()
parser.add_argument("--agent",            type=str, default=AGENT_GATED_ATTENTTION)
parser.add_argument("--dir-tensorboard",  type=str, default=".tensorboard")
parser.add_argument("--dir-checkpoints",  type=str, default=".checkpoints")
parser.add_argument("--benchmark-trials", type=int, default=10)
parser.add_argument("--benchmark-every",  type=int, default=10000)
parser.add_argument("--checkpoint-every", type=int, default=10000)
parser.add_argument("--seed",             type=int, default=0)
args = parser.parse_args()

# Reproducibility
fix_random_seeds(args.seed)

# Training instructions
training_instructions = [("Go to the green", [0]), ("Go to the red", [1]), ("Go to the blue", [2])]
instruction_tokenizer = InstructionTokenizer([instr[0] for instr in training_instructions])

# Agents and Environments
# Different agents need different information.
if args.agent == AGENT_GATED_ATTENTTION:
    agent    = GatedAttentionAgent(instruction_tokenizer)
    env_type = FindItemsEnvObsOnlyGrid
elif args.agent == AGENT_PERFECT:
    # Since the perfect agent should know everything, we provided it with the full information.
    # (such as the coordinates of the agent's position)
    agent    = PerfectAgent()
    env_type = FindItemsEnv
else:
    raise NotImplementedError("There is no such agent.")


env_definition = InstructionEnvironmentDefinition(
                                       env_type, width=10, height=10,
                                       num_items=3, must_avoid_non_targets=True,
                                       reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM)

writer = SummaryWriter(os.path.join(args.dir_tensorboard, env_definition.name(), agent.name()))

# Define benchmarks
trajectory_length_benchmark = SuccessTrajectoryLengthBenchmark(
    env_definition, n_trials_per_instruction=args.benchmark_trials
)
success_rate_benchmark      = SuccessRateBenchmark(
    env_definition, n_trials_per_instruction=args.benchmark_trials
)

agent.log_init(writer)
agent.train_init(env_definition, training_instructions)
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
print("Mean benchmark trajectory length: {}".format(trajectory_length_benchmark(agent, training_instructions)))
print("Mean success rate: {}".format(success_rate_benchmark(agent, training_instructions)))

# Show how the agent is behaving
env                  = env_definition.build_env(training_instructions[0])
obs, reward, done, _ = env.reset()
observations         = [obs]
agent.reset()
while not done:
    obs, reward, done, _ = env.step(agent.act(obs, env))
    observations.append(obs)
FindItemsVisualizator.pyplot_animate(observations)