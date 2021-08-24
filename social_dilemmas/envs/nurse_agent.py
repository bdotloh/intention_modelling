import copy
import random
import sys

from social_dilemmas.envs.agent import Agent
from social_dilemmas.envs.agent import BASE_ACTIONS
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

NURSE_ACTIONS = BASE_ACTIONS.copy()
NURSE_VIEW_SIZE = 7
MAX_ASTAR_DEPTH = 200
DISTANCE_FACTOR = 1
TIME_COST_FACTOR = 1


class NurseAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, view_len=NURSE_VIEW_SIZE):
        self.view_len = view_len
        # self.urgency = dict(urgency)
        # self.reward = dict(reward)
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return NURSE_ACTIONS[action_number]

    def get_done(self):
        return False

    def complete(self, char, pos_dict):
        """Defines how an agent interacts with the char it is standing on"""
        if char in pos_dict:
            return ' '
        else:
            return char

    def get_maze(self):
        grid_height = self.grid.shape[0]
        grid_width = self.grid.shape[1]
        maze = []
        for row_elem in range(grid_height):
            row = []
            for column_elem in range(grid_width):
                if self.grid[row_elem][column_elem] == '@':
                    row.append(1)
                else:
                    row.append(0)
            maze.append(row)
        return maze

    def dist_to_one_goal(self, x, y, goalx, goaly):
        """
        Calculates the distance to a goal located in (goalx, goaly)
        step1: Creates the maze of 0s and 1s where
            0 = allowed position of agent including the goals
            1 = not allowed position of agent e.g., walls
        step2: Calculates (using astar search) the number of steps to reach (goalx, goaly) position from agent's location (x, y)
        :param goalx: x coordinate of goal
        :param goaly: y coordinate of goal
        :return:
        Distance to goal (integer) if a path to goal is found
        Returns 100 (assume that it's very far) if a path to goal is not found
        """
        grid_height = self.grid.shape[0]
        grid_width = self.grid.shape[1]
        maze = []
        for row_elem in range(grid_height):
            row = []
            for column_elem in range(grid_width):
                if (self.grid[row_elem][column_elem] == ' ') or (self.grid[row_elem][column_elem] == 'B') or (self.grid[row_elem][column_elem] == 'S') or (self.grid[row_elem][column_elem] == 'T'):
                    row.append(0)
                elif self.grid[row_elem][column_elem] == '@':
                    row.append(1)
            maze.append(row)
        # TODO: need to update self.pos[0] to x and self.pos[1] to y so that other agents can also use this method, and check why there is error in astar.
        if goalx != float('inf'):
            path = astar(maze, (x, y), (goalx, goaly))
            if path != None:
                return len(path)
            else:
                return 100

    # find the level 0 goal of the current agent
    # obs - dictionary of possible goals {(x_coor, y_coor): 'goal_type'}
    # x, y - starting position
    # goal is calculated according to reward + urgency - distance
    def find_goal(self, goals_locs, x, y):
        """

        step1: Calculate item_cost using (distance between agent x and goal x) + (distance between agent y and goal y) divided by 10
        step2: If Urgency = True, it's added to utility as 1, else, add 0.

        :param goals_locs: dictionary of possible goals {(1, 2): 'B', (1, 5): 'B', (5, 5): 'B', (11, 5): 'B'}
        :param x:
        :param y:
        :return:
        """
        goal = [float('inf'), float('inf')]
        # print('goal_locs', goals_locs)
        # print('self.urgency', self.urgency)
        # print('self.reward', self.reward)
        for item in goals_locs:
            # print('item', item)
            item_cost = abs(x - item[0]) + abs(y - item[1])
            goal_cost = abs(x - goal[0]) + abs(y - goal[1])
            item_util = self.urgency[goals_locs[item]] + self.reward[goals_locs[item]] - item_cost / 10
            # print('item_util', item_util)
            if goal != [float('inf'), float('inf')]:
                goal_util = self.reward[goals_locs[tuple(goal)]] - goal_cost / 10
            else:
                goal_util = float('-inf')
            if item_util > goal_util:
                goal[0] = item[0]
                goal[1] = item[1]
            print('goal_util', goal_util)

        return goal

    def find_final_goal(self, x, y, goals_locs, agent_locs, depth):
        """
        Recursive function:
        step1: Get the distance of current agent position to each goal
        step2: For multiple agents, also find final goal. If the final goal distance of other agents is smaller compared to current agent, then abandon that goal
        step3: Utility function calculates the maximum utility: reward - goal distance/10  (this is 0.1 per step)
        :param x: agent's start x position
        :param y: agent's start y position
        :param goals_locs: dictionary of possible goals {(3, 2): 'B', (11, 5): 'B'}
        :param agent_locs: list of agent locations  e.g., [(7, 4)]
        :param depth: recursion depth
        :return:
        """
        if depth == 0:
            goal = self.find_goal(goals_locs, x, y)
            return goal

        else:
            # Initialize with my distance to all possible goals
            goal_dist = {goal: self.dist_to_one_goal(x, y, goal[0], goal[1]) for goal in goals_locs}
            for item in agent_locs:
                other_x = item[0]
                other_y = item[1]
                other_goal = self.find_final_goal(other_x, other_y, goals_locs, agent_locs, depth - 1)
                if other_goal == None:
                    other_dist = float('inf')
                else:
                    other_dist = self.dist_to_one_goal(other_x, other_y, other_goal[0], other_goal[1])
                if other_goal != None and other_goal != [float('inf'), float('inf')]:
                    # If other is closer to goal than I am
                    if other_dist < goal_dist[(other_goal[0], other_goal[1])]:
                        # Then goal is unreachable by me
                        goal_dist[(other_goal[0], other_goal[1])] = float('inf')
            # utility function reward - cost
            goal_util = {goal: self.reward[goals_locs[goal]] - goal_dist[goal] / 10 for goal in goal_dist}
            if goal_util != {}:
                goal = max(goal_util, key=goal_util.get)
                return goal if goal_util[goal] > 0 else None
            else:
                return None

    # x, y: goal coordinate
    def determine_action(self, x, y):
        """
        Determines action to the goal located in (x, y)
        :param x: x coordinate of goal
        :param y: y coordinate of goal
        :return:
        Action:  3 - go right; 2 - go left; 1 - go down; 0 - go up;
        """
        # get urgency
        urgent_goals = [goal for goal in self.urgency if self.urgency[goal]]
        # maze marking with 1 and 0: 1-not urgent, 0-urgent
        maze = []
        for row_elem in range(self.grid.shape[0]):
            row = []
            for column_elem in range(self.grid.shape[1]):
                if self.grid[row_elem][column_elem] == ' ' or self.grid[row_elem][column_elem] in urgent_goals:
                    row.append(0)
                else:
                    row.append(1)
            maze.append(row)
        path = astar(maze, (self.pos[0], self.pos[1]), (x, y))
        if path != None:
            (x_coor, y_coor) = path[1]
        else:
            return 4  # 4 - stay

        if (y_coor != self.pos[1]):
            action = 2 if self.pos[1] > y_coor else 3
        else:
            action = 0 if self.pos[0] > x_coor else 1
        return action
        # 3- go right; 2 - go left; 1 - gdo down; 0 - go up;

    def compute_distance_cost_from_agent(self, depth, goal_pos, distance_factor=DISTANCE_FACTOR):
        # distances = []
        paths_to_this_goal_label = []

        agent_x = self.pos[0]
        agent_y = self.pos[1]
        for single_goal_pos in goal_pos:
            goal_x = single_goal_pos[0]
            goal_y = single_goal_pos[1]
            all_paths_to_a_goal_loc = astar_allpaths(self.get_maze(), (agent_x, agent_y), (goal_x, goal_y), search_depth=depth)
            # print('to goal', single_goal_pos, 'astar_allpaths', all_paths_to_a_goal_loc)
            if all_paths_to_a_goal_loc:
                for path in all_paths_to_a_goal_loc:
                    paths_to_this_goal_label.append(path)
                    #distances.append(len(path) - 1)
                    #paths_to_this_goal_label.append(path)
        if paths_to_this_goal_label:
            return paths_to_this_goal_label
        return None  # very high cost if can't find the goal.

    def compute_distance_cost(self, depth, from_loc, to_loc, distance_factor=DISTANCE_FACTOR):
        distances = []
        paths_to_this_goal_label = []
        all_paths_to_a_goal_loc = astar_allpaths(self.get_maze(), from_loc, to_loc, search_depth=depth)
        if all_paths_to_a_goal_loc:
            for path in all_paths_to_a_goal_loc:
                distances.append(len(path) - 1)
                paths_to_this_goal_label.append(path)
        if distances:
            return min(distances)*distance_factor, paths_to_this_goal_label
        return None  # very high cost if can't find the goal.


    def compute_group_distance_cost(self, depth, subgoals, goals_pos):
        paths_to_high_level_goal = []
        # find the nearest goal from the agent location.
        least_distance_cost_from_agent_loc = float('inf')
        shortest_path_from_agent_loc = []
        nearest_goal_from_agent_loc = None

        for goal in subgoals:
            distance_cost, paths = self.compute_distance_cost_from_agent(depth, goals_pos[goal])
            #print('cost of subgoal {}: {}'.format(goal,distance_cost))
            if distance_cost < least_distance_cost_from_agent_loc:
                least_distance_cost_from_agent_loc = distance_cost
                shortest_path_from_agent_loc = paths
                nearest_goal_from_agent_loc = goal
            else:
                continue
        paths_to_high_level_goal.append(shortest_path_from_agent_loc)

        total_cost = float('inf')
        if nearest_goal_from_agent_loc:
            # go to other subgoals.
            copy_of_subgoals = copy.deepcopy(subgoals)
            copy_of_subgoals.remove(nearest_goal_from_agent_loc)
            #print('nearest_goal_from_agent_loc', nearest_goal_from_agent_loc)
            #print('goals_pos', goals_pos)
            from_loc_x = goals_pos[nearest_goal_from_agent_loc][0][0]
            from_loc_y = goals_pos[nearest_goal_from_agent_loc][0][1]

            total_cost = least_distance_cost_from_agent_loc
            while True:
                if copy_of_subgoals == []:
                    break

                min_cost = float('inf')
                for goal in copy_of_subgoals:
                    to_loc_x = goals_pos[goal][0][0]
                    to_loc_y = goals_pos[goal][0][1]
                    cost, path = self.compute_distance_cost(depth, (from_loc_x, from_loc_y), (to_loc_x, to_loc_y))
                    # print('goal', goal, copy_of_subgoals, to_loc_x, to_loc_y, from_loc_x, from_loc_y)
                    # print('cost', cost, 'path', path)
                    if cost < min_cost:
                        min_cost = cost
                        prev_goal = goal
                        prev_path = path
                from_loc_x = goals_pos[prev_goal][0][0]
                from_loc_y = goals_pos[prev_goal][0][1]
                total_cost += min_cost
                paths_to_high_level_goal.append(prev_path)
                copy_of_subgoals.remove(prev_goal)
            # print('total_cost', total_cost, copy_of_subgoals)
        return total_cost, paths_to_high_level_goal

    def compute_group_reward(self, goals_scores, subgoals):
        total_reward = 0
        for goal in subgoals:
            total_reward += goals_scores[goal]
        return total_reward

    def which_action(self, next_coord):

        # Action:  3 - go right; 2 - go left; 1 - go down; 0 - go up;
        # I have not checked the code but the social dilemma framework uses above Action id.
        x_coor = next_coord[0]
        y_coor = next_coord[1]

        if (y_coor != self.pos[1]):
            action = 2 if self.pos[1] > y_coor else 3
        elif (x_coor != self.pos[0]):
            action = 0 if self.pos[0] > x_coor else 1
        else:
            action = 4
        return action

    def get_available_goals(self, goals_pos):
        # print('goals_pos', goals_pos)
        # {'a': [[1, 4]], 'b': [[2, 2]], 'c': [[4, 3]], 'd': [[5, 5]], 'e': [[5, 8]], 'f': [], 'g': [], 'h': [], 'k': [], 'l': [], 'm': [], 'n': [], 'S': [[1, 10]], 'T': []}
        available_subgoals = []
        for goal_label, goal_locs in goals_pos.items():
            for goal_loc in goal_locs:
                if self.grid[goal_loc[0]][goal_loc[1]] == goal_label:
                    #print('goal {} is available'.format(goal_label))
                    available_subgoals.append(goal_label)
        '''            
        target_subgoals = [subgoal for subgoal in copy_of_goalgroups[self.target] if subgoal in available_subgoals]
        if target_subgoals:
            print('gonna get subgoals from goal {}'.format(self.target))
        if not target_subgoals:
            self.target = 'two'
            print('collected all subgoals from first target goal, moving on to goal {}'.format(self.target))

        target_subgoals = [subgoal for subgoal in copy_of_goalgroups[self.target] if subgoal in available_subgoals]
        #self.available_subgoals = target_subgoals
        print('subgoals in this goal: {}'.format(target_subgoals))
        return target_subgoals
        '''
        return available_subgoals

    def policy(self, depth, goal_groups, goals_scores, goals_pos,use_distance_cost=True, use_time_cost=False, time_factor=TIME_COST_FACTOR):
        ################################ work in progress ################################
        # utility = reward - time_cost - distance_cost
        # in this implementation, time cost is assumed to be accounted for in the reward (decaying reward)
        # print(self.grid)

        available_subgoals = self.get_available_goals(goals_pos)
        copy_of_goalgroups = copy.deepcopy(goal_groups)
        print(available_subgoals)
        if not available_subgoals:
            sys.exit('all goals collected :)')

        # checks whether goal is available
        for goal_group, goal_group_attributes in goal_groups.items():
            for goal,goal_attributes in goal_group_attributes['goals'].items():
                if goal not in available_subgoals:
                    goal_attributes['taken'] = True
                    goal_group_attributes['activated'] = True

            if goal_group_attributes['activated']:
                print(goal_group)

        # checks activation of all goal groups (i.e. whether any goal in goal_group taken)
        # if all goal groups unactivated, go for nearest obtainable goal (i.e. no prereqs)

        if not all(any([goal_groups[goal]['activated'] for goal in goal_groups.keys()]) for _ in range(len(goal_groups))):
            print('no goals activated')
            least_cost_from_agent_loc = float('inf')
            shortest_path = []
            nearest_goal = None
            for goal_group, goal_group_attributes in goal_groups.items():
                for goal, goal_attributes in goal_group_attributes['goals'].items():
                    if (goal_attributes['obtainable']) and not goal_attributes['taken']:
                        goal_paths = self.compute_distance_cost_from_agent(depth, goals_pos[goal])
                        print(goal,goal_paths)
                        if goal_paths:
                            distance_cost = len(goal_paths[0]) - 1
                            if distance_cost < least_cost_from_agent_loc:
                                least_cost_from_agent_loc = distance_cost
                                shortest_paths_from_agent_loc = goal_paths
                                nearest_goal_from_agent_loc = goal
                            else:
                                continue

            #print(nearest_goal_from_agent_loc,shortest_paths_from_agent_loc)
            action = self.which_action(shortest_paths_from_agent_loc[0][1])
        else:
            action = 4


        # goals_utility_dict = {}
        # for goal_group, goal_group_attributes in copy_of_goalgroups.items():
        #     for goal,goal_attributes in goal_group_attributes['t'].items():
        #         if goal not in available_subgoals:
        #             goal_attributes['taken'] = True
        #             goal_group_attributes['activated'] = True
        #         elif goal in available_subgoals and goal_attributes['obtainable'] == True:
        #             goal_attributes['taken'] = False
        #             goal_paths = self.compute_distance_cost_from_agent(depth,goals_pos[goal])
        #             if goal_paths:
        #                 print('goal: {}'.format(goal))
        #                 for i,path in enumerate(goal_paths):
        #                     print('path {} to goal {}'.format(i,goal))
        #                     for coordinates in path[1:]:
        #                         print(coordinates)


                    # if paths_to_goal:
                    #     #print(paths_to_goal[0])
                    #     if len(paths_to_goal[0]) > 1:
                    #         cost_path_dict[goal] = (cost, paths_to_goal)
            if goal_group_attributes['activated']:
                print(goal_group)

        print(copy_of_goalgroups)
        # if not cost_path_dict:
        #     sys.exit('no obtainable goals....')
        # target_goal = (min(cost_path_dict,key=cost_path_dict.get))
        # paths_to_target_subgoal = cost_path_dict[target_goal][-1][0]
        # action = self.which_action(paths_to_target_subgoal[1])
        # cost_path_dict.clear()

            # action = random.randint(0,3)


        '''
        for goal,subgoals in goal_groups.items():
            target_goal = goal
            subgoals_in_target_goal = [subgoal for subgoal in copy_of_goalgroups[target_goal] if subgoal in available_subgoals]
            if subgoals_in_target_goal:
                break
            else:
                continue

        target_subgoal = subgoals_in_target_goal[-1]  #target subgoal is last item in target goal's subgoal list
        print('collecting subgoal {} from target goal {}'.format(target_goal,target_subgoal))
        cost, paths_to_target_subgoal = self.compute_distance_cost_from_agent(depth, goals_pos[target_subgoal])
        for subgoal in available_subgoals:
            if subgoal is target_subgoal:
                continue
            else:
                for path in paths_to_target_subgoal:
                    if tuple(goals_pos[subgoal][0]) in path:
                        paths_to_target_subgoal.remove(path)
                        print('subgoal {} in path {} towards target subgoal, removing path...'.format(subgoal,path))
        
        if len(paths_to_target_subgoal[0]) > 1:
            action = self.which_action(paths_to_target_subgoal[0][1])
        else:
            action = 4
            # action = random.randint(0,3)
        '''
        return action


# astar search algorithm
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar_allpaths(maze, start, end, search_depth=MAX_ASTAR_DEPTH):
    '''
    Desmond modification: return _all_ optimal paths from the given start to the given end in the given maze
    '''

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    all_paths = []  ### ---- modification to astar_allpaths
    FOUND_FIRST = 0  ### ---- modification to astar_allpaths
    BEST_SCORE = 0  ### ---- modification to astar_allpaths

    iterations = 0
    # Loop until you find the end
    while len(open_list) > 0:
        iterations += 1
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            ### ---- BEGIN: modification to astar_allpaths
            if FOUND_FIRST == 0:
                FOUND_FIRST = 1
                BEST_SCORE = len(path[::-1])
                all_paths.append(path[::-1])
            elif len(path[::-1]) <= BEST_SCORE:
                all_paths.append(path[::-1])
            else:
                # found a new path, and length of path is longer than best path; return
                return all_paths
            # return path[::-1]  # Return reversed path
            ### ---- END: modification to astar_allpaths
        if iterations >= search_depth:
            if len(all_paths) > 0:
                return all_paths
            return None

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = (abs(child.position[0] - end_node.position[0])) + (abs(child.position[1] - end_node.position[1]))
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    iterations = 0
    # Loop until you find the end
    while len(open_list) > 0:
        iterations += 1
        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path
        if iterations >= MAX_ASTAR_DEPTH:
            return None

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = (abs(child.position[0] - end_node.position[0])) + (abs(child.position[1] - end_node.position[1]))
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

