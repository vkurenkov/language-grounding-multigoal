import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

class Grid:
    def __init__(self, width, height, num_items=3):
        '''
        num_items - number of item types that can be presented on the grid
        '''
        self._num_items = num_items
        self._width = width
        self._height = height
        self._grid = np.zeros(shape=(num_items, width, height))

    def _check_grid_range(self, x, y):
        if not self.is_within_the_grid(x, y):
            raise Exception("Given position is out of grid.")

    def is_within_the_grid(self, x, y):
        if x < 0 or x >= self._width:
            return False
        if y < 0 or y >= self._height:
            return False

        return True

    def clear(self):
        self._grid.fill(0)
    
    def mark_at(self, x, y, item=0):
        self._check_grid_range(x, y)
        self._grid[item, x, y] = 1

    def unmark_at(self, x, y, item=0):
        self._check_grid_range(x, y)
        self._grid[item, x, y] = 0

    def get_grid(self, copy=False):
        if copy:
            return np.copy(self._grid)
        else:
            return self._grid

    def get_items_at(self, x, y):
        self._check_grid_range(x, y)
        return self._grid[:, x, y]


class FindItemsEnv(gym.Env):
    '''
    The agent is instructed to find given items on the grid.
    The agent must visit all of the specified items in the proper order.

    Agent:
        - Actions: Move Forward; Move Backwards; Move Left; Move Right; Turn Right; Turn Left;
        - Has a look direction (one of the neighborhood cells)

    Reward:
        - Can be of types of sparsity:
            1 - Get reward for reaching only the last item.
            2 - Get reward for reaching every item.
        - Can be shaped
            1 - Reward - is a manhattan distance to the current item + reward for looking in the right direction.

    Observation:
        - The entire grid of size (num_items, width, height)
        - Or agent's view (num_items, side, side), it is a square
        
        - Items are enumerated starting from the zero
        - The agent's current position (not included in the grid)
    '''

    LOOK_WEST = "WEST"
    LOOK_EAST = "EAST"
    LOOK_NORTH = "NORTH"
    LOOK_SOUTH = "SOUTH"

    ACTION_MOVE_FORWARD = 0
    ACTION_MOVE_BACKWARDS = 1
    ACTION_MOVE_LEFT = 2
    ACTION_MOVE_RIGHT = 3
    ACTION_TURN_LEFT = 4
    ACTION_TURN_RIGHT = 5

    NO_OBJECT = -1

    REWARD_TYPE_MIN_ACTIONS = 0
    REWARD_TYPE_EVERY_ITEM = 1
    REWARD_TYPE_LAST_ITEM = 2
    
    def __init__(self, width, height, num_items, reward_type,
                 instruction=[0], must_avoid_non_targets=False,
                 fixed_positions=None, fixed_look=None):
        '''
        num_items - number of unique items on the grid
                    no more than this number of items will be placed on the grid
        reward_type - what kind of reward shaping is applied
        must_avoid_non_targets - whether the agent must avoid non current target objects
        instruction - an array of items that must be visited in the specified order
                e.g [0, 2, 1] - first the agent must visit 0, then 2, then 1
        fixed_positions - an array of (x, y) positions to place an agent and items
                        - first element is for the agent
                        - must be of size num_items + 1
        fixed_look - WEST | EAST | NORTH | SOUTH
        '''
        self._num_items = num_items
        self._grid = Grid(width, height, num_items)
        self._reward_type = reward_type
        self._must_avoid_non_targets = must_avoid_non_targets

        if fixed_positions is not None and len(fixed_positions) != (self._num_items + 1):
            raise Exception("Number of fixed positions must be equal to number of items + 1 (for the agent).")
        if fixed_positions is not None and fixed_look is None:
            raise Exception("Fixed positions and fixed look must be specified alltogether!")
        if fixed_look is not None and fixed_positions is None:
            raise Exception("Fixed positions and fixed look must be specified alltogether!")

        self._fixed_positions = fixed_positions
        self._fixed_look = fixed_look

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({
            "agent_position": gym.spaces.Tuple((gym.spaces.Discrete(width), gym.spaces.Discrete(height))),
            "agent_look": gym.spaces.Discrete(4),
            "grid": gym.spaces.Box(0.0, 1.0, shape=self._grid.get_grid().shape, dtype=np.float32)
        })

        self.instruction = instruction
        self.reset()

    def _reset_instruction(self, instruction):
        self.instruction = instruction
        self._items_visit_order = instruction
        self._items_visit_order_pos = []

        # Fill in positions of items to visit
        for item in self._items_visit_order:
            self._items_visit_order_pos.append(self._items_pos[item])

    def _get_distance(self, pos0, pos1):
        return abs(pos0[0] - pos1[0]) + abs(pos0[1] - pos1[1])

    def _min_num_actions_to_target(self):
        if self._has_done():
            return 0

        # For both cases: num_moves + num_rotations to face the object
        # And we should check for cells around the target
        if self._must_avoid_non_targets:
            raise NotImplementedError()
        else:
            target_pos = self._get_target_pos()
            agent_pos = self._agent_pos
            agent_look_dir = self._agent_look_dir()

            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            neighbors = [(target_pos[0] + dir[0], target_pos[1] + dir[1]) for dir in dirs]
            neighbors = [pos for pos in neighbors if self._grid.is_within_the_grid(pos[0], pos[1])]

            min_actions = 1000000000 # Do you think it's enough? Won't work for very big maps ;(
            for pos in neighbors:
                # Calculate number of steps
                num_steps = self._get_distance(agent_pos, pos)

                # Calculate number of rotations
                num_rotations = 0
                look_pos = (pos[0] + agent_look_dir[0], pos[1] + agent_look_dir[1])
                if(look_pos != target_pos):
                    # Draw a pic, it'll get easier to understand
                    if look_pos[0] == target_pos[0] or look_pos[1] == target_pos[1]:
                        num_rotations = 2
                    else:
                        num_rotations = 1

                if min_actions > num_rotations + num_steps:
                    min_actions = num_rotations + num_steps

            return -min_actions

    def _randomly_place_items_and_agent(self):
        free_cells = [(x, y) for x in range(self._grid._width) for y in range(self._grid._height)]
        chosen_indices = np.random.choice(len(free_cells), size=self._num_items + 1, replace=False)

        if self._fixed_positions is not None:
            chosen_cells = self._fixed_positions
        else:
            chosen_cells = [free_cells[index] for index in chosen_indices]

        # Randomly place and rotate the agent
        self._place_agent_at(chosen_cells[0][0], chosen_cells[0][1])

        if self._fixed_look is not None:
            self._look_agent_at(str.upper(self._fixed_look))
        else:
            self._look_agent_at(np.random.choice(["WEST", "EAST", "NORTH", "SOUTH"]))

        # Spawn other items
        self._items_pos = {}
        for item, cell in enumerate(chosen_cells[1:]):
            self._items_pos[item] = cell
            self._spawn_item_at(cell[0], cell[1], item)

    def _place_agent_at(self, x, y):
        self._agent_pos = (x, y)

    def _look_agent_at(self, side):
        '''
        input:
            side - WEST|EAST|NORTH|SOUTH
        '''
        self._agent_look = str.upper(side)

    def _turn_agent(self, left=True):
        if left:
            if self._agent_look == FindItemsEnv.LOOK_NORTH:
                self._look_agent_at(FindItemsEnv.LOOK_WEST)
            elif self._agent_look == FindItemsEnv.LOOK_WEST:
                self._look_agent_at(FindItemsEnv.LOOK_SOUTH)
            elif self._agent_look == FindItemsEnv.LOOK_SOUTH:
                self._look_agent_at(FindItemsEnv.LOOK_EAST)
            elif self._agent_look == FindItemsEnv.LOOK_EAST:
                self._look_agent_at(FindItemsEnv.LOOK_NORTH)
            else:
                raise Exception("No such look direction")
        else:
            if self._agent_look == FindItemsEnv.LOOK_NORTH:
                self._look_agent_at(FindItemsEnv.LOOK_EAST)
            elif self._agent_look == FindItemsEnv.LOOK_WEST:
                self._look_agent_at(FindItemsEnv.LOOK_NORTH)
            elif self._agent_look == FindItemsEnv.LOOK_SOUTH:
                self._look_agent_at(FindItemsEnv.LOOK_WEST)
            elif self._agent_look == FindItemsEnv.LOOK_EAST:
                self._look_agent_at(FindItemsEnv.LOOK_SOUTH)
            else:
                raise Exception("No such look direction")

    def _agent_look_dir(self):
        return FindItemsEnv.look_to_dir(self._agent_look)
    
    def _agent_sees(self):
        '''
        output:
            item - type of the item the agent looks at right now (defaults to FindItemsEnv.NO_OBJECT)
        '''
        look_dir = self._agent_look_dir()
        look_pos = (self._agent_pos[0] + look_dir[0], self._agent_pos[1] + look_dir[1])

        if not self._grid.is_within_the_grid(look_pos[0], look_pos[1]):
            return FindItemsEnv.NO_OBJECT
        else:
            items = self._grid.get_items_at(look_pos[0], look_pos[1])
            if not items.any():
                return FindItemsEnv.NO_OBJECT
            else:
                return np.argmax(items)

    def _cur_target_item(self):
        if len(self._items_visit_order) == len(self._visited_items):
            return FindItemsEnv.NO_OBJECT
        else:
            return self._items_visit_order[len(self._visited_items)]

    def _get_target_pos(self):
        target_ind = len(self._visited_items)
        if target_ind == self._items_visit_order:
            return self._items_visit_order_pos[target_ind - 1]
        else:
            return self._items_visit_order_pos[target_ind]

    def _spawn_item_at(self, x, y, item):
        self._grid.mark_at(x, y, item=item)

    def _add_to_visited(self, item):
        if item != FindItemsEnv.NO_OBJECT:
            self._visited_items.append(item)

    def _update_visited_items(self):
        observed_item = self._agent_sees()
        self._add_to_visited(observed_item)

    def _current_observation(self):
        return (self._agent_pos, self._agent_look, self._grid.get_grid())

    def _current_reward(self):
        if(self._reward_type == FindItemsEnv.REWARD_TYPE_MIN_ACTIONS):
            return self._min_num_actions_to_target()
        elif(self._reward_type == FindItemsEnv.REWARD_TYPE_EVERY_ITEM):
            if self._cur_target_item() == self._agent_sees():
                return 1.0
        elif(self._reward_type == FindItemsEnv.REWARD_TYPE_LAST_ITEM):
            if len(self._items_visit_order) == len(self._visited_items) + 1:
                if self._cur_target_item() == self._agent_sees():
                    return 1.0
        else:
            raise Exception("No such reward type!")
        
        return 0.0

    def _has_done(self):
        target_item = self._cur_target_item()
        observed_item = self._agent_sees()
        # All items were seen
        if target_item == FindItemsEnv.NO_OBJECT:
            return True
        else:
            if observed_item == FindItemsEnv.NO_OBJECT:
                return False
            else:
                if observed_item != target_item:
                    if self._must_avoid_non_targets:
                        return True
                    else:
                        return False
                else:
                    if len(self._items_visit_order) == len(self._visited_items) + 1:
                        return True

    def _gym_output(self):
        '''
        output:
            observation - an observation after the action, tuple (agent_pos, agent_dir, grid)
            reward - a scalar reward after the action was done
            done - whether the mission is terminated (successful or not)
            info - defaults to None
        '''
        return self._current_observation(), self._current_reward(), self._has_done(), None

    @staticmethod
    def look_to_dir(look):
        if look == FindItemsEnv.LOOK_EAST:
            return (1, 0)
        elif look == FindItemsEnv.LOOK_WEST:
            return (-1, 0)
        elif look == FindItemsEnv.LOOK_NORTH:
            return (0, 1)
        elif look == FindItemsEnv.LOOK_SOUTH:
            return (0, -1)

        raise Exception("No such look direction") 

    def handle_message(self, msg):
        """
        Handles custom messages (useful for asynchronous environments).
        This one assumes that all messages are instructions to be set.
        """
        self._reset_instruction(msg)

    def reset(self):
        '''
        output:
            observation - an observation after the action, tuple (agent_pos, agent_dir, grid)
            reward - a scalar reward after the action was done
            done - whether the mission is terminated (successful or not)
            info - defaults to None
        '''
        self._grid.clear()
        self._randomly_place_items_and_agent()
        self._reset_instruction(self.instruction)
        self._visited_items = []

        # First obtain current observation
        output = self._gym_output()

        # Then update visited items
        self._update_visited_items()
        
        return output

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action):
        '''
        input:
            action: 0 - Move Forward; 1 - Move Backward; 2 - Move Left; 3 - Move Right; 4 - Turn Left; 5 - Turn Right;
        output:
            observation - an observation after the action, tuple (agent_pos, agent_dir, grid)
            reward - a scalar reward after the action was done
            done - whether the mission is terminated (successful or not)
        '''
        if self._has_done():
            return self._gym_output()
            
        # Act
        if action == FindItemsEnv.ACTION_TURN_LEFT:
            self._turn_agent(left=True)
        elif action == FindItemsEnv.ACTION_TURN_RIGHT:
            self._turn_agent(left=False)
        else:
            look_dir = self._agent_look_dir()
            agent_pos = self._agent_pos
            pos_to_move = None
            if action == FindItemsEnv.ACTION_MOVE_FORWARD:
                pos_to_move = (agent_pos[0] + look_dir[0], agent_pos[1] + look_dir[1])
            elif action == FindItemsEnv.ACTION_MOVE_BACKWARDS:
                pos_to_move = (agent_pos[0] - look_dir[0], agent_pos[1] - look_dir[1])
            elif action == FindItemsEnv.ACTION_MOVE_RIGHT:
                if look_dir[1] != 0:
                    pos_to_move = (agent_pos[0] + look_dir[1], agent_pos[1] + look_dir[0])
                else:
                    pos_to_move = (agent_pos[0] - look_dir[1], agent_pos[1] - look_dir[0])
            elif action == FindItemsEnv.ACTION_MOVE_LEFT:
                if look_dir[1] != 0:
                    pos_to_move = (agent_pos[0] - look_dir[1], agent_pos[1] - look_dir[0])
                else:
                    pos_to_move = (agent_pos[0] + look_dir[1], agent_pos[1] + look_dir[0])
            else:
                raise Exception("There is no such action.")

            if self._grid.is_within_the_grid(pos_to_move[0], pos_to_move[1]):
                self._place_agent_at(pos_to_move[0], pos_to_move[1])

        # First - calculate rewards, state, and so on
        output = self._gym_output()

        # Then update list of visited items
        self._update_visited_items()

        return output

    def name(self):
        obs_type = "full_obs"
        if self._fixed_positions is None:
            placement = "randomized"
        else:
            positions = "_".join([f"({x}_{y})" for (x, y) in self._fixed_positions])
            placement = "fixed_" + positions + f"_{self._fixed_look}"

        grid = "grid_" + str(self._grid._width) + "_" + str(self._grid._height) + "_" + str(self._num_items)
        instruction = "instr_" + "_".join([str(item) for item in self.instruction])

        rew_type = "rew_unknown"
        if self._reward_type == self.REWARD_TYPE_MIN_ACTIONS:
            rew_type = "rew_min_action"
        elif self._reward_type == self.REWARD_TYPE_EVERY_ITEM:
            rew_type = "rew_every_item"
        elif self._reward_type == self.REWARD_TYPE_LAST_ITEM:
            rew_type = "rew_last_item"

        return f"gridworld/{obs_type}/{placement}/{grid}/{instruction}/{rew_type}/"

class FindItemsVisualizator:
    @staticmethod
    def pyplot(env):
        fig, a = plt.subplots()
        
        # Plot the grid
        grid = env._grid
        num_items = grid._num_items
        width = grid._width
        height = grid._height

        a.set_xlim((0, width))
        a.set_ylim((0, height))
        for x in range(width):
            for y in range(height):
                # Draw a grid cell
                rect = plt.Rectangle((x, y), 1, 1, fill=False)
                a.add_artist(rect)
            
                # Draw an item at the cell
                items = grid.get_items_at(x, y)
                if items.any():
                    item = np.argmax(items)

                    cmap = plt.cm.rainbow
                    norm = plt_colors.Normalize(0, num_items)
                    itm = plt.Rectangle((x + 0.25, y + 0.25), 0.5, 0.5, color=cmap(norm(item)))
                    txt = plt.Text(x + 0.3, y + 0.3, text=str(item))
                    a.add_artist(itm)
                    a.add_artist(txt)

        # Plot the agent position
        agent_pos = (env._agent_pos[0] + 0.5, env._agent_pos[1] + 0.5)
        agent = plt.Circle(agent_pos, radius=0.25, color="r")
        a.add_artist(agent)

        # Plot the aget's look direction
        agent_dir = env._agent_look_dir()
        agent_look = plt.Line2D([agent_pos[0], agent_pos[0] + agent_dir[0]],
                                [agent_pos[1], agent_pos[1] + agent_dir[1]],
                                color="b")
        a.add_artist(agent_look)

        plt.show(fig)


if __name__ == "__main__":
    # Grid: Test Grid Marking
    grid = Grid(10, 10)
    grid.mark_at(9, 9, item=1)
    assert(grid.get_items_at(9, 9)[1] == 1)
    print("Grid: Mark grid cell - Success.")

    # Grid: Test Grid Unmarking
    grid = Grid(10, 10)
    grid.mark_at(9, 9, item=1)
    grid.unmark_at(9, 9, item=1)
    assert(grid.get_items_at(9, 9)[1] == 0)
    print("Grid: Unmark grid cell - Success.")

    # Grid: Test Grid Clear
    grid = Grid(10, 10)
    grid.mark_at(9, 9, item=1)
    grid.clear()
    assert(grid.get_grid().all() == 0)
    print("Grid: Clear grid - Success.")

    # Grid: Test Get Grid Copy
    grid = Grid(10, 10)
    grid.mark_at(9, 9, item=1)
    copied_grid = grid.get_grid(copy=True)
    grid.unmark_at(9, 9, item=1)
    assert(copied_grid[1, 9, 9] == 1)
    print("Grid: Get copied grid - Success.")

    # Grid: Test Get Grid no copy
    grid = Grid(10, 10)
    grid.mark_at(9, 9, item=1)
    copied_grid = grid.get_grid(copy=True)
    grid.unmark_at(9, 9, item=1)
    assert(copied_grid[1, 9, 9] == 1)
    print("Grid: Get original grid - Success.")

    # Grid: Test Grid Range checker
    grid = Grid(10, 10)
    try:
        grid.mark_at(9, 10, item=1)
    except:
        assert(True)
        print("Grid: Grid size check - Success.")
    else:
        assert(False)

    # FindItemsEnv: Test Seed Reproducibility
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    obs, reward, done, _ = env.reset([0, 2, 1])

    env.seed(0)
    obs1, reward1, done1, _ = env.reset([0, 2, 1])

    env1 = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env1.seed(0)
    obs2, reward2, done2, _ = env1.reset([0, 2, 1])

    assert(obs1[0] == obs[0])
    assert(obs1[1] == obs[1])
    assert(np.array_equal(obs1[2], obs[2]))
    assert(reward1 == reward)
    assert(done1 == done)


    assert(obs2[0] == obs[0])
    assert(obs2[1] == obs[1])
    assert(np.array_equal(obs2[2], obs[2]))
    assert(done2 == done)
    print("FindItemsEnv: Test Seed Reproducibility - Success.")

    # FindItemsEnv: Agent must move forward
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (2, 6)
    agent_look = obs[1] # North

    obs1, _, _, _ = env.step(0) # Move forward

    agent_pos1 = obs1[0]
    assert(agent_pos1 == (agent_pos[0], agent_pos[1] + 1))
    print("FindItemsEnv: Agent must move forward - Success.")
    
    # FindItemsEnv: Agent must move backwards
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (2, 6)
    agent_look = obs[1] # North

    obs1, _, _, _ = env.step(1) # Move backwards

    agent_pos1 = obs1[0]
    assert(agent_pos1 == (agent_pos[0], agent_pos[1] - 1))
    print("FindItemsEnv: Agent must move backwards - Success.")

    # FindItemsEnv: Agent must move left
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (2, 6)
    agent_look = obs[1] # North

    obs1, _, _, _ = env.step(2) # Move left

    agent_pos1 = obs1[0]
    assert(agent_pos1 == (agent_pos[0] - 1, agent_pos[1]))
    print("FindItemsEnv: Agent must move left - Success.")

    # FindItemsEnv: Agent must move right
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (2, 6)
    agent_look = obs[1] # North

    obs1, _, _, _ = env.step(3) # Move right

    agent_pos1 = obs1[0]
    assert(agent_pos1 == (agent_pos[0] + 1, agent_pos[1]))
    print("FindItemsEnv: Agent must move right - Success.")

    # FindItemsEnv: Agent must turn left
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (2, 6)
    agent_look = obs[1] # North

    obs1, _, _, _ = env.step(4) # Turn left

    agent_look1 = obs1[1]
    assert(agent_look1 == "WEST")
    print("FindItemsEnv: Agent must turn left - Success.")

    # FindItemsEnv: Agent must turn right
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (2, 6)
    agent_look = obs[1] # North

    obs1, _, _, _ = env.step(5) # Turn right

    agent_look1 = obs1[1]
    assert(agent_look1 == "EAST")
    print("FindItemsEnv: Agent must turn right - Success.")

    # FindItemsEnv: Agent must stay if it's blocked moving forward
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(1)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (8, 0)

    env.step(FindItemsEnv.ACTION_TURN_RIGHT)
    env.step(FindItemsEnv.ACTION_TURN_RIGHT)
    obs, _, _, _ = env.step(FindItemsEnv.ACTION_MOVE_FORWARD)
    agent_pos1 = obs[0]
    assert(agent_pos == agent_pos1)
    print("FindItemsEnv: Agent must stay if it's blocked moving forward - Success.")

    # FindItemsEnv: Agent must stay if it's blocked moving backwards
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(1)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (8, 0)

    obs, _, _, _ = env.step(FindItemsEnv.ACTION_MOVE_BACKWARDS)
    agent_pos1 = obs[0]
    assert(agent_pos == agent_pos1)
    print("FindItemsEnv: Agent must stay if it's blocked moving backwards - Success.")

    # FindItemsEnv: Agent must stay if it's blocked moving left
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(1)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (8, 0)

    env.step(FindItemsEnv.ACTION_TURN_LEFT)
    obs, _, _, _ = env.step(FindItemsEnv.ACTION_MOVE_LEFT)
    agent_pos1 = obs[0]
    assert(agent_pos == agent_pos1)
    print("FindItemsEnv: Agent must stay if it's blocked moving left - Success.")

    # FindItemsEnv: Agent must stay if it's blocked moving right
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(1)
    obs, _, _, _ = env.reset([0, 1, 2])
    agent_pos = obs[0] # (8, 0)

    env.step(FindItemsEnv.ACTION_TURN_RIGHT)
    obs, _, _, _ = env.step(FindItemsEnv.ACTION_MOVE_RIGHT)
    agent_pos1 = obs[0]
    assert(agent_pos == agent_pos1)
    print("FindItemsEnv: Agent must stay if it's blocked moving right - Success.")

    # FindItemsEnv: Environment must end if the agent observes a non-target object if specified
    env = FindItemsEnv(10, 10, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM, True)
    env.seed(1)
    obs, reward, done, _ = env.reset([0, 1, 2])
    assert(done)
    print("FindItemsEnv: Environment must end if the agent observes a non-target object if specified - Success.")

    # FindItemsEnv: Environment must end if the agent observes the last target object
    env = FindItemsEnv(5, 5, 3, FindItemsEnv.REWARD_TYPE_LAST_ITEM)
    env.seed(0)
    env.reset([0])

    env.step(FindItemsEnv.ACTION_MOVE_RIGHT)
    obs, reward, done, _ = env.step(FindItemsEnv.ACTION_MOVE_RIGHT)

    assert(reward == 1.0 and done)
    print("FindItemsEnv: Environment must end if the agent observes the last target object - Success.")

    # FindItemsEnv: Agent must get a reward for observing the target object
    env = FindItemsEnv(5, 5, 3, FindItemsEnv.REWARD_TYPE_EVERY_ITEM)
    env.seed(0)
    env.reset([0, 1, 2])

    env.step(FindItemsEnv.ACTION_MOVE_RIGHT)
    obs, reward, _, _ = env.step(FindItemsEnv.ACTION_MOVE_RIGHT)

    assert(reward == 1.0)
    print("FindItemsEnv: Agent must get a reward for observing the target object - Success.")

    # FindItemsEnv: Agent must get a reward for observing the last target object
    env = FindItemsEnv(5, 5, 3, FindItemsEnv.REWARD_TYPE_LAST_ITEM)
    env.seed(0)
    env.reset([0])

    env.step(FindItemsEnv.ACTION_MOVE_RIGHT)
    obs, reward, _, _ = env.step(FindItemsEnv.ACTION_MOVE_RIGHT)

    assert(reward == 1.0)
    print("FindItemsEnv: Agent must get a reward for observing the last target object - Success.")

    # FindItemsEnv: Agent must get a reward proportional to the distance to the target object
    env = FindItemsEnv(5, 5, 3, FindItemsEnv.REWARD_TYPE_MIN_ACTIONS)
    env.seed(0)
    obs, reward, _, _ = env.reset([1])

    assert(reward == -6) # 6 actions to reach the item
    print("FindItemsEnv: Agent must get a reward proportional to the distance to the target object - Success.")

    # FindItemsEnv: Environment must provide a partial view if specified
