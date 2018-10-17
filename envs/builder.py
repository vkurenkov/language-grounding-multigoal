from envs.gridworld.env import FindItemsEnv
from utils.gym import ObservationStack, SubprocVecEnv, LimitedSteps


def build_async_environment(environment="grid_world", stack_size=4, num_async_envs=10, **kwargs):
    '''
    params:
        environment - ["grid_world"]
        stack_size - how many frames to concatenate
        num_async_envs - how many environments in parallel
    '''
    if environment != "grid_world":
        raise NotImplementedError()

    return SubprocVecEnv([ObservationStack(LimitedSteps(FindItemsEnv(**kwargs)), stack_size) for _ in range(num_async_envs)])
