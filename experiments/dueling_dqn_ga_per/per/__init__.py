"""
Taken from here: https://github.com/rlcode/per/blob/master/
"""

import random
import numpy as np

from experiments.dueling_dqn_ga_per.per.sumtree import SumTree

class PrioritizedProportionalReplay:
    # To avoid zero probabilities
    e    = 0.01

    # Smothing the priority (0.0 - uniform; 1.0 - proportional to error)
    a    = 1.0

    # Bias-correction via importance-sampling (is)
    # The closer to 1.0 the proper the 'is' weight becomes
    beta = 1.0
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity=1000000):
        self.tree     = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def append(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch      = []
        idxs       = []
        segment    = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Imortance Sampling weights for bias-correction
        #sampling_probabilities = priorities / self.tree.total()
        #is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        #is_weight /= is_weight.max()

        return batch, idxs

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)