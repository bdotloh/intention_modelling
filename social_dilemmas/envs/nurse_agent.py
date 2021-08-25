import copy
import random
import sys

from social_dilemmas.envs.agent import Agent
from social_dilemmas.envs.agent import BASE_ACTIONS
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

import torch
from torch.nn import Softmax
from itertools import permutations

#NURSE_ACTIONS = BASE_ACTIONS.copy()
NURSE_VIEW_SIZE = 7
MAX_ASTAR_DEPTH = 2000
DISTANCE_FACTOR = 1
TIME_COST_FACTOR = 1


class NurseAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, state_beliefs, view_len=NURSE_VIEW_SIZE):
        self.view_len = view_len
        # self.urgency = dict(urgency)
        # self.reward = dict(reward)
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)
        self.state_beliefs = state_beliefs
        self.softmax = Softmax(dim=-1)
        self.ix_to_act = BASE_ACTIONS.copy()
        self.possessions = []

    @property
    def action_space(self):
        return Discrete(9)

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.view_len + 1,
                                             2 * self.view_len + 1, 3), dtype=np.float32)

    def get_done(self):
        return False

    def complete(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char in self.state_beliefs.keys():
            if not self.state_beliefs[char]['requires']:
                self.possessions.append(char)
                return ' '
            else:
                return char
        else:
            return char

    def compute_reward(self, char, goals_score):
        print('CHAR',char)
        print('GOALS_DICT',goals_score)
        if char in goals_score.keys():

            reward_this_turn = goals_score[char]
        else:
            reward_this_turn = 0

        return reward_this_turn

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

    def compute_distance_cost_from_agent(self, depth, goal, distance_factor=DISTANCE_FACTOR):
        distances = []
        paths_to_this_goal_label = []

        agent_x = self.pos[0]
        agent_y = self.pos[1]

        goal_x = self.state_beliefs[goal]['location'][0]
        goal_y = self.state_beliefs[goal]['location'][-1]

        all_paths_to_a_goal_loc = astar_allpaths(self.get_maze(), (agent_x, agent_y), (goal_x, goal_y),
                                                 search_depth=depth)
        # print('to goal', single_goal_pos, 'astar_allpaths', all_paths_to_a_goal_loc)
        if all_paths_to_a_goal_loc:
            for path in all_paths_to_a_goal_loc:
                paths_to_this_goal_label.append(path)
                distances.append(len(path) - 1)
        if distances:
            return min(distances) * distance_factor, paths_to_this_goal_label
        return 1000, None  # very high cost if can't find the goal.

    def compute_distance_cost(self, depth, from_loc, to_loc, distance_factor=DISTANCE_FACTOR):
        distances = []
        paths_to_this_goal_label = []
        all_paths_to_a_goal_loc = astar_allpaths(self.get_maze(), from_loc, to_loc, search_depth=depth)
        if all_paths_to_a_goal_loc:
            for path in all_paths_to_a_goal_loc:
                distances.append(len(path) - 1)
                paths_to_this_goal_label.append(path)
        if distances:
            return min(distances) * distance_factor, paths_to_this_goal_label
        return 1000, None  # very high cost if can't find the goal.

    def compute_intention_cost(self, intention, state_beliefs):
        """
        intention: ordered list of available goals
        Check state beliefs for whether there are any urgent goals in intentions
        and retrieve index of final urgent goal in intention.
        Compute cost from agent's current location to first goal in intention
        Compute cost of moving from one goal to another by iterating over intention list
        Cost per step is 5x more if urgent goal is not yet completed.

        e.g.
        Intention = [a,b,c,d]
        urgent_goal = b
        cost of moving from:
        agent's current pos to a: 5 * distance cost
        a to b: 5 * distance cost
        b to c: distance cost
        c to d: distance cost
        """
        intention_cost = 0
        path_to_intention = []
        urgent_goal = None   #store urgent goals in list
        for goal, beliefs in state_beliefs.items():
            if beliefs['urgency']:
                urgent_goal = goal

        if not urgent_goal:
            pass
        elif urgent_goal:
            index_of_final_urgent_goal_in_intention = max([goal_index for goal_index in range(len(intention)) if intention[goal_index] in urgent_goal])

        cost, paths = self.compute_distance_cost_from_agent(MAX_ASTAR_DEPTH, intention[0])
        path_to_intention.extend(random.choice(paths)[1:])
        if urgent_goal:
            cost = cost * 3
        intention_cost += cost
        for i, goal in enumerate(intention):
            if i < len(intention)-1:
                from_loc = self.state_beliefs[intention[i]]['location']
                to_loc = self.state_beliefs[intention[i+1]]['location']
                cost, paths = self.compute_distance_cost(MAX_ASTAR_DEPTH, from_loc, to_loc)
                path_to_intention.extend(random.choice(paths)[1:])
                if urgent_goal:
                    if i + 1 <= index_of_final_urgent_goal_in_intention:
                        cost = cost * 3
                intention_cost += cost

        return intention_cost, path_to_intention

    def compute_intention_reward(self, intention, state_beliefs):
        intentions_reward = 0
        sequential_goal = None
        for goal, beliefs in state_beliefs.items():
            if beliefs['requires']:
                sequential_goal = goal

        if sequential_goal:
            prerequisite_goals = [goal for goal in self.state_beliefs[sequential_goal]['requires']]
            index_of_final_prerequisite_goal_in_intention = max([intention.index(goal) for goal in prerequisite_goals])

            for i, goal in enumerate(intention):
                if goal == sequential_goal:
                    if i < index_of_final_prerequisite_goal_in_intention:
                        intentions_reward += 0
                    else:
                        intentions_reward += 100
                else:
                    intentions_reward+=100

        else:
            intentions_reward = len(intention) * 100

        return intentions_reward

    def sample_intention_given_utility(self, intention_space, state_beliefs):
        unnormalised_utilities = torch.zeros(len(intention_space))
        paths = []

        for i, intention in enumerate(intention_space):
            reward = self.compute_intention_reward(intention,state_beliefs)
            cost, path = self.compute_intention_cost(intention, state_beliefs)
            utility = reward - cost
            unnormalised_utilities[i] = utility
            paths.append(path)
            #print('unnorm({}) = {} - {} = {}'.format(intention,reward,cost,utility))

        normalised_utilities = self.softmax(unnormalised_utilities)

        for i, intention in enumerate(intention_space):
            print('normalised U({}) = {}'.format(intention, normalised_utilities[i]))

        intention_distribution = torch.distributions.Categorical(normalised_utilities)
        sampled_intention_ix = intention_distribution.sample().item()

        sampled_intention = intention_space[sampled_intention_ix]
        path_of_sampled_intention = paths[sampled_intention_ix]

        #print('$$$$sampled intention {} $$$$'.format(sampled_intention))

        return path_of_sampled_intention

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
        return self.ix_to_act[action]

    def get_available_goals(self):
        available_goals = []
        for goal, beliefs in self.state_beliefs.items():
            goal_loc = beliefs['location']
            if self.grid[goal_loc[0]][goal_loc[1]] == goal:
                # print('goal {} is available'.format(goal_label))
                available_goals.append(goal)
        return available_goals

    def policy(self):
        state_beliefs = {goal: self.state_beliefs[goal] for goal in self.get_available_goals()}
        for beliefs in state_beliefs.values():
            if beliefs['requires']:
                prereq_goals = beliefs['requires']
                for goal in prereq_goals:
                    if goal not in state_beliefs.keys():
                        beliefs['requires'].remove(goal)

        print('state beliefs', state_beliefs)


        if state_beliefs:
            intention_space = []
            goal_permutations = permutations(state_beliefs.keys())
            for goal_permutation in list(goal_permutations):
                goal_permutation = list(goal_permutation)
                intention_space.append(goal_permutation)

            path_of_sampled_intention = self.sample_intention_given_utility(intention_space, state_beliefs)
            action = self.which_action(path_of_sampled_intention[0])
        else:
            action = 'STAY'
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
