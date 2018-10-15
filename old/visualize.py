from drawnow import drawnow
from utils.training import Session

import random
import time
import matplotlib.pyplot as plt
import numpy as np

session = Session("agent-1-trial-0").start()
session.switch_group(0)

rewards = session.get_rewards()[:, 0]
timesteps = session.get_timesteps()[:, 0]

ticks = []
mean_rewards = []
mean_timesteps = []

n = 200
start = 0
end = start + n
while end < len(rewards):
    mean_rewards.append(np.mean(rewards[start:end]))
    mean_timesteps.append(np.mean(timesteps[start:end]))

    if len(ticks) > 0:
        ticks.append(ticks[len(ticks) - 1] + n)
    else:
        ticks.append(n)

    start = end
    end = start + n

plt.plot(mean_timesteps)
plt.show()