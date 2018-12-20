import gym
import enum


class GoalStatus(enum.Enum):
    SUCCESS     = 1
    FAILURE     = 2
    IN_PROGRESS = 3


class GoalEnv(gym.Env):
    def goal_status(self) -> GoalStatus:
        raise NotImplementedError()