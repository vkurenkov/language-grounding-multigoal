from envs.gridworld.env import FindItemsEnv
from utils.gym import ObservationStack, SubprocVecEnv, LimitedSteps, RewardMinMaxScaler


REWARD_SCALERS = ["none", "min_max"]
ENVIRONMENTS = ["grid_world"]


def build_async_environment(environment="grid_world", reward_scaler={"name": "none"},
                            stack_size=4, num_async_envs=10, **kwargs):
    '''
    params:
        environment - ["grid_world"]
        stack_size - how many frames to concatenate
        num_async_envs - how many environments in parallel
    '''
    if environment not in ENVIRONMENTS:
        raise NotImplementedError("This environment is not implemented.")
    if reward_scaler["name"] not in REWARD_SCALERS:
        raise NotImplementedError("This reward scaling strategy is not implemented.")

    if reward_scaler["name"] == "min_max":
        min_reward = reward_scaler["min_reward"]
        max_reward = reward_scaler["max_reward"]
        return SubprocVecEnv([
            ObservationStack(RewardMinMaxScaler(LimitedSteps(FindItemsEnv(**kwargs), 40), min_reward, max_reward), stack_size)
            for _ in range(num_async_envs)
        ])
    else:
        return SubprocVecEnv([
            ObservationStack(LimitedSteps(FindItemsEnv(**kwargs), 40), stack_size)
            for _ in range(num_async_envs)
        ])
