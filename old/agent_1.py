from env import FindItemsEnv
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from utils.training import Session

import os
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import numpy.random as np_rand


class QValue(nn.Module):
    def __init__(self, n_features, n_actions=8):
        super(QValue, self).__init__()

        self.linear = nn.Linear(n_features, n_actions)

    def forward(self, features):
        '''
        input:
            features - of size (batch_size, n_features)
        output:
            action_values - estimated action values of size (batch_size, n_actions)
        '''
        return self.linear(features)


class Observation:
    @staticmethod
    def to_features(observation, env):
        look_dir = env._agent_look_dir()
        agent_pos = observation[0]
        grid = observation[2]

        feature_agent_pos = np.array([agent_pos[0] / env._grid._width, agent_pos[1] / env._grid._height])
        feature_agent_look = np.array([look_dir[0], look_dir[1]])
        feature_grid = grid.flatten()

        return np.concatenate([feature_agent_pos, feature_agent_look, feature_grid]).reshape(1, -1)

    @staticmethod
    def num_features():
        return 304

    @staticmethod
    def to_torch(features, volatile=False):
        return Variable(FloatTensor(features), volatile=volatile)


class Replay:
    def __init__(self):
        self._replay = []

    def append(mementos):
        self._replay.extend(memento)

    def sample(n):
        if n > len(self._replay):
            return np_rand.choice(self._replay, size=len(self._replay), replace=False)
        else:
            return np_rand.choice(self._replay, size=n, replace=False)


# Constants
NUM_ACTIONS = 6

EPSILON_START = 0.95
EPSILON_FINISH = 0.01
FRAMES = 1000000
DISCOUNT = 0.95

REPORT_EPISODES = 50
CHECKPOINT_FRAMES = 50000

NUM_TRIALS = 20
seed = -1

for trial in range(NUM_TRIALS):
    with Session("agent-1-1-trial-" + str(trial)).start() as sess:
        sess.switch_group()
        
        # Log
        sess.log("Trial #" + str(trial) + " started.\n")

        # Networks
        behaviour_q = QValue(Observation.num_features(), NUM_ACTIONS)
        optimizer = torch.optim.SGD(behaviour_q.parameters(), lr=0.01)

        # Environment
        env = FindItemsEnv(10, 10, 3, reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS)
        num_frames = 0
        last_report = 0
        last_checkpoint = 0
        epsilon = EPSILON_START
        timesteps = []

        while num_frames < FRAMES:
            target_item = 0

            # Set simulation seed
            seed += 1
            env.seed(seed)
            np_rand.seed(seed)
            torch.manual_seed(seed)

            # Retrieve first observation
            obs, reward, done, _ = env.reset([target_item])
            features = Observation.to_features(obs, env)

            # Track current trajectory
            trajectory = []
            num_timesteps = 0

            while not done:
                # Select an action in an epsilon-greedy way
                action_values = behaviour_q.forward(Observation.to_torch(features, volatile=True))
                action = np_rand.random_integers(0, NUM_ACTIONS-1)
                if np_rand.rand() > epsilon:
                    action = torch.max(action_values, dim=-1)[1].data[0]

                # Take an action and keep current observation
                obs, reward, done, _ = env.step(action)
                features_after = Observation.to_features(obs, env)
                trajectory.append((features, action, reward, features_after))

                # Update behaviour network online
                point = trajectory[-1]
                point_action = Variable(LongTensor([[int(point[1])]]))
                point_reward = Variable(FloatTensor([[point[2]]]))
                point_features_after = Observation.to_torch(point[3], volatile=True)

                estimated_values = behaviour_q.forward(Observation.to_torch(point[0]))
                estimated_values += torch.max(Variable(behaviour_q.forward(point_features_after).data)) * DISCOUNT

                loss = (point_reward - torch.gather(estimated_values, dim=-1, index=point_action))** 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update current observation
                features = features_after

                # Update epsilon and number of frames
                num_frames += 1
                num_timesteps += 1
                epsilon = EPSILON_START - (EPSILON_START - EPSILON_FINISH) * (num_frames / FRAMES)

            sess.timesteps(num_timesteps)
            sess.reward(num_frames) # No hidden meaning here, just to retrieve further
            timesteps = sess.get_timesteps()

            # Report current progress
            if len(timesteps) % REPORT_EPISODES == 0:
                averaged = np.mean(timesteps[-REPORT_EPISODES:])

                sess.log("Current progress: " + str(round((num_frames / FRAMES) * 100, 2)) + "%")
                sess.log("\n")
                sess.log("Current epsilon: " + str(epsilon))
                sess.log("\n")
                sess.log("Last 50 episodes averaged: " + str(averaged))
                sess.log("\n")
                sess.log("Frames seen: " + str(num_frames))
                sess.log("\n")

                if averaged <= 10:
                    sess.log("Success!!!")
                    sess.log("\n")
                    sess.checkpoint_model(behaviour_q, "model_" + str(num_frames) + ".pt")
                    sess.log("Saved the model: " + str(num_frames))
                    sess.log("\n")
                    break

                last_report = last_report
                sess.log_flush()


            # Save current network
            if num_frames - last_checkpoint >= CHECKPOINT_FRAMES:
                sess.checkpoint_model(behaviour_q, "model_" + str(num_frames) + ".pt")
                sess.log("Saved the model: " + str(num_frames))
                sess.log("\n")
                sess.log_flush()
                last_checkpoint = num_frames