import numpy as np
from envs.gridworld_simple.env import FindItemsEnv, FindItemsVisualizator
from envs.goal.env import GoalStatus

# Seed 0; 3 objects
# # # # # # # # # # # # 
# - - - - - - - - - - #
# - - - - - - - - - - #
# - - - - - - - - - - #
# - - A - - - - - 0 - #
# - - - - - 2 - - - - #
# - - - - - - - - - - #
# - - - - - - - - - - #
# 1 - - - - - - - - - #
# - - - - - - - - - - #
# - - - - - - - - - - #
# # # # # # # # # # # # 


def test_env_should_support_requirement_of_avoiding_non_target_objects():
    env = FindItemsEnv(10, 10, 3, 
            reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM,
            instruction=[1, 0, 2], 
            must_avoid_non_targets=True)
    env.seed(0)
    env.reset()

    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    obs, rew, done, _ = env.step(env.ACTION_MOVE_DOWN)

    # The agent has reached a wrong object
    assert (rew == 0 and done)


def test_env_should_support_fixing_items_and_agent_positions():
    env = FindItemsEnv(10, 10, 3,
            reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM,
            instruction=[1, 0, 2],
            must_avoid_non_targets=True,
            fixed_positions=[(1, 2), (5, 5), (3, 3), (2, 2)])
    env.seed(0)

    (agent_pos, grid), _, _, _ = env.reset()

    # All positions were fixed
    zero_at_5_5 = np.array_equal(np.ravel(np.nonzero(grid[0, :])), [5, 5])
    one_at_3_3  = np.array_equal(np.ravel(np.nonzero(grid[1, :])), [3, 3])
    two_at_2_2  = np.array_equal(np.ravel(np.nonzero(grid[2, :])), [2, 2])

    assert (agent_pos == (1, 2) and zero_at_5_5 and one_at_3_3 and two_at_2_2)


def test_env_should_support_multiple_items():
    env = FindItemsEnv(10, 10, 3,
            reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM,
            instruction=[1, 0, 2],
            must_avoid_non_targets=True)
    env.seed(0)

    (_, grid), _, _, _ = env.reset()

    assert grid.shape[0] == 4 


def test_env_should_support_reward_as_minimum_number_of_actions_when_avoiding_non_targets():
    # Seed: 1; Instruction: [0, 1, 2]
    # # # # # # # # # # # # 
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 0 - #
    # - - - 1 - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 2 - #
    # - - - - - - - - A - #
    # # # # # # # # # # # # 
    env = FindItemsEnv(10, 10, 3,
            reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS,
            instruction=[0, 1, 2],
            must_avoid_non_targets=True)

    # At the beginning
    env.seed(1)
    _, reward, *_ = env.reset()
    assert reward == 0

    # Arriving at 0
    _, reward, *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_UP)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_UP)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_UP)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_UP)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_LEFT)
    assert reward == 1

    # Arriving at 1
    _, reward, *_ = env.step(env.ACTION_MOVE_LEFT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_LEFT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_LEFT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_LEFT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_LEFT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 1

    # Arriving at 2
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert reward == 1
    _, reward, *_ = env.step(env.ACTION_MOVE_UP)
    assert reward == -1
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 1
    assert env.goal_status() == GoalStatus.IN_PROGRESS

    _, reward, *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert reward == 11
    assert env.goal_status() == GoalStatus.SUCCESS

    _, reward, *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert reward == -10
    assert env.goal_status() == GoalStatus.SUCCESS


def test_env_should_support_reward_as_minimum_number_of_actions_when_not_avoiding_non_targets():
    # Seed: 1; Instruction: [0, 1, 2]
    # # # # # # # # # # # # 
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 0 - #
    # - - - 1 - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 2 - #
    # - - - - - - - - A - #
    # # # # # # # # # # # # 
    env = FindItemsEnv(10, 10, 3,
            reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS,
            instruction=[0, 1, 2],
            must_avoid_non_targets=False)

    # At the beginning
    env.seed(1)
    _, reward, *_ = env.reset()
    assert reward == 0

    # After the first item
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    _, reward, *_ = env.step(env.ACTION_MOVE_UP)

    assert reward == 1

    # Finishing
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 1

    # To be sure
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 1


def test_env_should_support_reward_per_every_item_in_instruction():
    env = FindItemsEnv(10, 10, 3, 
            reward_type=FindItemsEnv.REWARD_TYPE_EVERY_ITEM,
            instruction=[1, 0, 2], 
            must_avoid_non_targets=True)
    env.seed(0)
    obs, *_ = env.reset()

    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_LEFT)
    obs, rew, done, _ = env.step(env.ACTION_MOVE_LEFT)
    
    # The agent has reached the first object
    assert (rew == 1 and not done)


def test_env_should_support_reward_for_last_item_in_instruction():
    env = FindItemsEnv(10, 10, 3, 
            reward_type=FindItemsEnv.REWARD_TYPE_LAST_ITEM,
            instruction=[1, 0, 2], 
            must_avoid_non_targets=True)
    env.seed(0)
    env.reset()

    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_DOWN)
    env.step(env.ACTION_MOVE_LEFT)
    obs, rew, done, _ = env.step(env.ACTION_MOVE_LEFT)
    
    # The agent has reached the first object
    assert (rew == 0 and not done)

    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    obs, rew, done, _ = env.step(env.ACTION_MOVE_UP)

    # The agent has reached the second object
    assert (rew == 0 and not done)

    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    obs, rew, done, _ = env.step(env.ACTION_MOVE_DOWN)

    # The agent has reached the final object
    assert (rew == 1 and done)


def test_env_should_output_grid_and_agent_position():
    env = FindItemsEnv(10, 10, 3, 
            reward_type=FindItemsEnv.REWARD_TYPE_LAST_ITEM,
            instruction=[1, 0, 2], 
            must_avoid_non_targets=True,
            fixed_positions=[(0, 0), (2, 2), (3, 3), (4, 4)])
    env.seed(0)
    obs, *_ = env.reset()

    agent_pos = obs[0]
    grid      = obs[1]

    assert (agent_pos == (0, 0)) and (grid.shape == (4, 10, 10))


def test_env_should_allow_agent_to_move_in_4_directions():
    env = FindItemsEnv(10, 10, 3, 
            reward_type=FindItemsEnv.REWARD_TYPE_LAST_ITEM,
            instruction=[1, 0, 2], 
            must_avoid_non_targets=True,
            fixed_positions=[(8, 8), (2, 2), (3, 3), (4, 4)])
    env.seed(0)
    env.reset()

    # Move left
    (agent_pos, _), *_ = env.step(env.ACTION_MOVE_LEFT)
    assert agent_pos == (7, 8)

    # Move right
    (agent_pos, _), *_ = env.step(env.ACTION_MOVE_RIGHT)
    assert agent_pos == (8, 8)

    # Move down
    (agent_pos, _), *_ = env.step(env.ACTION_MOVE_DOWN)
    assert agent_pos == (8, 7)

    # Move up
    (agent_pos, _), *_ = env.step(env.ACTION_MOVE_UP)
    assert agent_pos == (8, 8)
    

def test_env_should_track_goal_status_when_not_avoiding_non_target_objects():
    # Seed: 1; Instruction: [0, 1, 2]
    # # # # # # # # # # # # 
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 0 - #
    # - - - 1 - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 2 - #
    # - - - - - - - - A - #
    # # # # # # # # # # # # 
    env = FindItemsEnv(10, 10, 3,
            reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS,
            instruction=[0, 1, 2],
            must_avoid_non_targets=False)

    # At the beginning
    env.seed(1)
    _, reward, *_ = env.reset()
    assert env.goal_status() == GoalStatus.IN_PROGRESS

    # After the first item
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    _, reward, *_ = env.step(env.ACTION_MOVE_UP)

    assert env.goal_status() == GoalStatus.IN_PROGRESS

    # At the second item
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert env.goal_status() == GoalStatus.IN_PROGRESS

    # At the third item
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_DOWN)
    _, reward, done, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 11 and done and env.goal_status() == GoalStatus.SUCCESS


def test_env_should_track_goal_failure_when_avoiding_non_target_objects():
    # Seed: 1; Instruction: [0, 1, 2]
    # # # # # # # # # # # # 
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 0 - #
    # - - - 1 - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 2 - #
    # - - - - - - - - A - #
    # # # # # # # # # # # # 
    env = FindItemsEnv(10, 10, 3,
            reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS,
            instruction=[0, 1, 2],
            must_avoid_non_targets=True)

    # At the beginning
    env.seed(1)
    _, reward, *_ = env.reset()
    assert env.goal_status() == GoalStatus.IN_PROGRESS

    # Arrive at wrong object
    _, _, done, *_ = env.step(env.ACTION_MOVE_UP)
    assert done and env.goal_status() == GoalStatus.FAILURE


def test_env_should_track_goal_success_when_avoiding_non_target_objects():
    # Seed: 1; Instruction: [0, 1, 2]
    # # # # # # # # # # # # 
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 0 - #
    # - - - 1 - - - - - - #
    # - - - - - - - - - - #
    # - - - - - - - - 2 - #
    # - - - - - - - - A - #
    # # # # # # # # # # # # 
    env = FindItemsEnv(10, 10, 3,
            reward_type=FindItemsEnv.REWARD_TYPE_MIN_ACTIONS,
            instruction=[0, 1, 2],
            must_avoid_non_targets=True)

    # At the beginning
    env.seed(1)
    _, reward, *_ = env.reset()
    assert env.goal_status() == GoalStatus.IN_PROGRESS

    # After the first item
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    env.step(env.ACTION_MOVE_UP)
    _, reward, *_ = env.step(env.ACTION_MOVE_LEFT)

    assert env.goal_status() == GoalStatus.IN_PROGRESS

    # At the second item
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    env.step(env.ACTION_MOVE_LEFT)
    _, reward, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert env.goal_status() == GoalStatus.IN_PROGRESS

    # At the third item
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_RIGHT)
    env.step(env.ACTION_MOVE_DOWN)
    _, reward, done, *_ = env.step(env.ACTION_MOVE_DOWN)
    assert reward == 11 and done and env.goal_status() == GoalStatus.SUCCESS