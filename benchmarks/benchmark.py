import numpy as np
import warnings

from agents.agent import Agent
from envs.definitions import InstructionEnvironmentDefinition, Instruction
from envs.goal.env import GoalStatus
from typing import List


class Benchmark:
    def __init__(self, env_definition: InstructionEnvironmentDefinition, 
        n_trials_per_instruction=10, seed=0):
        self.n_trials_per_instruction = n_trials_per_instruction
        self.env_definition = env_definition
        self.seed = 0

    def __call__(self, agent: Agent, instructions: List[Instruction]):
        raise NotImplementedError()


class SuccessRateBenchmark(Benchmark):
    def __init__(self, env_definition: InstructionEnvironmentDefinition,
            n_trials_per_instruction=10, seed=0):
        return super().__init__(env_definition, n_trials_per_instruction, seed)

    def __call__(self, agent: Agent, instructions: List[Instruction]):
        success_rates = []
        for instruction in instructions:
            env = self.env_definition.build_env(instruction)
            # Benchmark every instruction with the same conditions
            env.seed(self.seed)

            n_success = 0
            for i in range(self.n_trials_per_instruction):
                obs, rew, done, _ = env.reset()
                while not done:
                    action = agent.act(obs, env)
                    if action:
                        obs, rew, done, _ = env.step(action)
                    else:
                        warnings.warn("An agent could not choose an action. Since the idle action is not supported, the trial is terminated.")
                        done = True

                    if env.goal_status() == GoalStatus.SUCCESS:
                        n_success += 1

            success_rates.append(n_success / self.n_trials_per_instruction)

        return np.mean(success_rates)


class SuccessTrajectoryLengthBenchmark(Benchmark):
    """
    Benchmarks length of successful trajectories.
    If a trajectory was not successfull, we assign maximum trajectory length (max_steps)
    """
    def __init__(self, env_definition: InstructionEnvironmentDefinition, 
            n_trials_per_instruction=10, seed=0, max_steps=100):
        self.max_steps = max_steps
        return super().__init__(env_definition, n_trials_per_instruction, seed)

    def __call__(self, agent: Agent, instructions: List[Instruction]):
        trajectory_lengths = []
        for instruction in instructions:
            env = self.env_definition.build_env(instruction)
            # Benchmark every instruction with the same conditions
            env.seed(self.seed)

            lengths = []
            for i in range(self.n_trials_per_instruction):
                obs, rew, done, _ = env.reset()
                n_steps = 0
                while not done and env.goal_status() == GoalStatus.IN_PROGRESS:
                    action = agent.act(obs, env)
                    if action:
                        obs, rew, done, _ = env.step(action)
                        n_steps += 1
                    else:
                        warnings.warn("An agent could not choose an action. Since the idle action is not supported, the episode is terminated with maximum number of steps.")
                        done = True
                        n_steps = self.max_steps

                if env.goal_status() == GoalStatus.FAILURE:
                    n_steps = self.max_steps

                lengths.append(n_steps)

            trajectory_lengths.append(np.mean(lengths))

        return np.mean(trajectory_lengths)