import gym
import enum


class GoalStatus(enum.Enum):
    SUCCESS     = 0
    FAILURE     = 1
    IN_PROGRESS = 2


class GoalEnv(gym.Env):
    def goal_status(self) -> GoalStatus:
        raise NotImplementedError()