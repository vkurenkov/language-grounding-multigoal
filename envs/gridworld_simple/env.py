import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import heapq

from typing import Tuple, List, Dict, Optional
from envs.goal.env import GoalEnv, GoalStatus


# Type Aliases
Position  = Tuple[int, int]
Path      = List[Position]
ItemType  = int


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


class FindItemsEnv(GoalEnv):
    '''
    The agent is instructed to find given items on the grid.
    The agent must visit all of the specified items in the proper order.

    Agent:
        - Actions: Move Up; Move Down; Move Left; Move Right;

    Reward:
        - Can be of types of sparsity:
            1 - Get reward for reaching only the last item.
            2 - Get reward for reaching every item.
        - Can be shaped
            1 - Reward - is a minimum manhattan distance to the current item.

    Observation:
        - The entire grid of size (num_items, width, height)
        - Or agent's view (num_items, side, side), it is a square
        
        - Items are enumerated starting from the zero
        - The agent's current position (not included in the grid)
    '''

    ACTION_MOVE_UP = 1
    ACTION_MOVE_DOWN = 2
    ACTION_MOVE_LEFT = 3
    ACTION_MOVE_RIGHT = 4

    NO_OBJECT = -1

    REWARD_TYPE_MIN_ACTIONS = 1
    REWARD_TYPE_EVERY_ITEM = 2
    REWARD_TYPE_LAST_ITEM = 3
    
    def __init__(self, width, height, num_items, reward_type,
                 instruction=[0], must_avoid_non_targets=False,
                 fixed_positions=None):
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
        '''
        self._num_items = num_items
        self._grid = Grid(width, height, num_items)
        self._reward_type = reward_type
        self._must_avoid_non_targets = must_avoid_non_targets

        if fixed_positions is not None and len(fixed_positions) != (self._num_items + 1):
            raise Exception("Number of fixed positions must be equal to number of items + 1 (for the agent).")

        self._fixed_positions = fixed_positions

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Dict({
            "agent_position": gym.spaces.Tuple((gym.spaces.Discrete(width), gym.spaces.Discrete(height))),
            "grid": gym.spaces.Box(0.0, 1.0, shape=self._grid.get_grid().shape, dtype=np.float32)
        })

        self.instruction = instruction
        self.reset()

    def _not_solvable(self):
        start_pos = self._agent_pos
        for cur_item in self._items_visit_order:
            if self._shortest_paths.get_path(start_pos, cur_item) is None:
                return True

            start_pos = self._items_pos[cur_item]

        return False

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
        # If it's done - it's done
        if self._has_done():
            return 0

        # One item or non-avoidance of the target objects degrades to a manhattan distance
        if self._num_items == 1 or not self._must_avoid_non_targets:
            return -self._get_distance(self._agent_pos, self._items_pos[self._cur_target_item()])
        else:
            return -len(self._shortest_paths.get_path(self._agent_pos, self._cur_target_item()))

    def _randomly_place_items_and_agent(self):
        free_cells = [(x, y) for x in range(self._grid._width) for y in range(self._grid._height)]
        chosen_indices = np.random.choice(len(free_cells), size=self._num_items + 1, replace=False)

        if self._fixed_positions is not None:
            chosen_cells = self._fixed_positions
        else:
            chosen_cells = [free_cells[index] for index in chosen_indices]

        # Randomly place and rotate the agent
        self._place_agent_at(chosen_cells[0][0], chosen_cells[0][1])

        # Spawn other items
        self._items_pos = {}
        for item, cell in enumerate(chosen_cells[1:]):
            self._items_pos[item] = cell
            self._spawn_item_at(cell[0], cell[1], item)

    def _place_agent_at(self, x, y):
        self._agent_pos = (x, y)
    
    def _agent_stands_at(self):
        '''
        output:
            item - type of the item the agent looks at right now (defaults to FindItemsEnv.NO_OBJECT)
        '''
        items = self._grid.get_items_at(self._agent_pos[0], self._agent_pos[1])
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
        observed_item = self._agent_stands_at()
        if not self._must_avoid_non_targets:
            if observed_item == self._cur_target_item():
                self._add_to_visited(observed_item)
        else:
            self._add_to_visited(observed_item)

    def _current_observation(self):
        return (self._agent_pos, self._grid.get_grid())

    def _current_reward(self):
        if(self._reward_type == FindItemsEnv.REWARD_TYPE_MIN_ACTIONS):
            return self._min_num_actions_to_target()
        elif(self._reward_type == FindItemsEnv.REWARD_TYPE_EVERY_ITEM):
            if self._cur_target_item() == self._agent_stands_at():
                return 1.0
        elif(self._reward_type == FindItemsEnv.REWARD_TYPE_LAST_ITEM):
            if len(self._items_visit_order) == len(self._visited_items) + 1:
                if self._cur_target_item() == self._agent_stands_at():
                    return 1.0
        else:
            raise Exception("No such reward type!")
        
        return 0.0

    def _has_done(self):
        target_item = self._cur_target_item()
        observed_item = self._agent_stands_at()
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
                    else:
                        return False

    def _gym_output(self):
        '''
        output:
            observation - an observation after the action, tuple (agent_pos, agent_dir, grid)
            reward - a scalar reward after the action was done
            done - whether the mission is terminated (successful or not)
            info - defaults to None
        '''
        return self._current_observation(), self._current_reward(), self._has_done(), None

    def handle_message(self, msg):
        """
        Handles custom messages (useful for asynchronous environments).
        This one assumes that all messages are instructions to be set.
        """
        self._reset_instruction(msg)

    def goal_status(self) -> GoalStatus:
        for i in range(len(self._visited_items)):
            if self._visited_items[i] != self._items_visit_order[i]:
                return GoalStatus.FAILURE
            if i >= (len(self._items_visit_order) - 1):
                return GoalStatus.SUCCESS
        
        return GoalStatus.IN_PROGRESS

    def reset(self):
        '''
        output:
            observation - an observation after the action, tuple (agent_pos, grid)
            reward - a scalar reward after the action was done
            done - whether the mission is terminated (successful or not)
            info - defaults to None
        '''
        self._grid.clear()
        self._randomly_place_items_and_agent()
        self._reset_instruction(self.instruction)
        self._visited_items = []

        # Build shortest paths to items from every point
        self._shortest_paths = FindItemsEnvShortestPaths(self)

        # Reset until the environment is solvable
        if self._not_solvable():
            return self.reset()

        # Obtain current observation
        output = self._gym_output()

        # Then update visited items
        self._update_visited_items()
        
        return output

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, action):
        '''
        input:
            action: 0 - Move Up; 1 - Move Down; 2 - Move Left; 3 - Move Right;
        output:
            observation - an observation after the action, tuple (agent_pos, grid)
            reward - a scalar reward after the action was done
            done - whether the mission is terminated (successful or not)
        '''
        agent_pos = self._agent_pos
        pos_to_move = None
        if action == FindItemsEnv.ACTION_MOVE_UP:
            pos_to_move = (agent_pos[0], agent_pos[1] + 1)
        elif action == FindItemsEnv.ACTION_MOVE_DOWN:
            pos_to_move = (agent_pos[0], agent_pos[1] - 1)
        elif action == FindItemsEnv.ACTION_MOVE_RIGHT:
            pos_to_move = (agent_pos[0] + 1, agent_pos[1])
        elif action == FindItemsEnv.ACTION_MOVE_LEFT:
            pos_to_move = (agent_pos[0] - 1, agent_pos[1])
        else:
            raise Exception("There is no such action.")

        if self._grid.is_within_the_grid(pos_to_move[0], pos_to_move[1]):
            self._place_agent_at(pos_to_move[0], pos_to_move[1])

        # First - calculate rewards, state, and so on
        output = self._gym_output()

        self._update_visited_items()

        return output

    def name(self):
        obs_type = "full_obs"
        if self._fixed_positions is None:
            placement = "randomized"
        else:
            positions = "_".join(["(" + str(x) + "_" + str(y) + ")" for (x, y) in self._fixed_positions])
            placement = "fixed_" + positions + "_" + str(self._fixed_look)

        grid = "grid_" + str(self._grid._width) + "_" + str(self._grid._height) + "_" + str(self._num_items)
        instruction = "instr_" + "_".join([str(item) for item in self.instruction])

        rew_type = "rew_unknown"
        if self._reward_type == self.REWARD_TYPE_MIN_ACTIONS:
            rew_type = "rew_min_action"
        elif self._reward_type == self.REWARD_TYPE_EVERY_ITEM:
            rew_type = "rew_every_item"
        elif self._reward_type == self.REWARD_TYPE_LAST_ITEM:
            rew_type = "rew_last_item"

        return "/".join(["gridworld", str(obs_type), str(placement), str(grid), str(instruction), str(rew_type)])


class FindItemsEnvShortestPaths:
    def __init__(self, env: FindItemsEnv):
        self._paths = {}
        self._env = env
        for x in range(env._grid._width):
            for y in range(env._grid._height):
                pos = (x, y)
                self._paths[pos] = self._find_shortest_paths(pos)

    def _find_shortest_paths(self, from_pos: Position) -> Dict[ItemType, Optional[Path]]:
        seen_vertices     = set([])
        work_vertices     = []
        from_pos_distance = {}
        previous_vertex   = {}

        # These are the positions that the agent cannot go from (except the starting pos)
        forbidden_positions = set([])
        if self._env._must_avoid_non_targets:
            forbidden_positions = set(self._env._items_pos.values())

        # Initialize Djikstra
        for x in range(self._env._grid._width):
            for y in range(self._env._grid._height):
                from_pos_distance[(x, y)] = np.inf
                previous_vertex[(x, y)]   = None

        heapq.heappush(work_vertices, (from_pos, 0))
        from_pos_distance[from_pos] = 0

        # Run Djikstra for the entire field
        while len(work_vertices) != 0:
            (cur_vertex, _)   = heapq.heappop(work_vertices)
            cur_distance      = from_pos_distance[cur_vertex]

            # We cannot go from the forbidden positions (except if it is a starting one)
            # But we can go to the forbidden positions
            if cur_vertex in forbidden_positions and cur_vertex != from_pos:
                continue

            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for neighbor in [(cur_vertex[0] + dx, cur_vertex[1] + dy) for (dx, dy) in directions]:
                if not self._env._grid.is_within_the_grid(neighbor[0], neighbor[1]):
                    continue

                alt_distance = cur_distance + 1
                if alt_distance < from_pos_distance[neighbor]:
                    from_pos_distance[neighbor] = alt_distance
                    previous_vertex[neighbor]   = cur_vertex

                    if neighbor not in seen_vertices:
                        heapq.heappush(work_vertices, (neighbor, alt_distance))
                        seen_vertices.add(neighbor)

        # Collect shortest paths for every item
        paths = {}
        for item in range(self._env._num_items):
            target_pos = self._env._items_pos[item]

            # Check if it's the starting position
            if target_pos == from_pos:
                paths[item] = []
            else:
                paths[item] = [target_pos]
                prev_pos    = previous_vertex[target_pos]
                while prev_pos != None and prev_pos != from_pos:
                    paths[item].append(prev_pos)
                    prev_pos    = previous_vertex[prev_pos]

                if prev_pos == None:
                    # There is no path from the source to the target position
                    paths[item] = None
                elif prev_pos == from_pos:
                    paths[item].reverse()

        return paths

    def get_path(self, from_pos: Position, to_item: ItemType) -> Optional[Path]:
        return self._paths[from_pos][to_item]


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

        plt.show(fig)

    @staticmethod
    def print(env):
        grid = env._grid
        num_items = grid._num_items
        width = grid._width
        height = grid._height

        print("# " * (width + 2))
        for y in reversed(range(height)):
            print("# ", end="")
            for x in range(width):
                character = "-"
                items = grid.get_items_at(x, y)
                if items.any():
                    character = str(np.argmax(items))
                if (x, y) == env._agent_pos:
                    character = "A"
                print(character, end=" ")
            print("#")
        print("# " * (width + 2))
