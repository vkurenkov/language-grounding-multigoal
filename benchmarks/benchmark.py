import numpy as np


class Benchmark:
    def __init__(self, env_definition, n_trials=10):
        self.n_trials = n_trials
        self.env_definition = env_definition

    def __call__(self, agent):
        raise NotImplementedError()


class SuccessRateBenchmark(Benchmark):
    def __init__(self, env_definition, n_trials=10):
        return super().__init__(env_definition, n_trials)

    def __call__(self, agent):
        env = self.env_definition.build_env()
        n_success = 0
        for i in range(self.n_trials):
            obs, rew, done, _ = env.reset()
            while not done:
                obs, rew, done, _ = env.step(agent.act(obs))
                if rew > 0.0 and done:
                    n_success += 1

        return n_success / self.n_trials


class TrajectoryLengthBenchmark(Benchmark):
    def __init__(self, env_definition, n_trials=10, max_steps=100):
        self.max_steps = max_steps
        return super().__init__(env_definition, n_trials)

    def __call__(self, agent):
        env = self.env_definition.build_env()
        lengths = []
        for i in range(self.n_trials):
            obs, rew, done, _ = env.reset()
            n_steps = 0
            while not done and n_steps < self.max_steps:
                obs, rew, done, _ = env.step(agent.act(obs))
                n_steps += 1

            lengths.append(n_steps)

        return np.mean(lengths)