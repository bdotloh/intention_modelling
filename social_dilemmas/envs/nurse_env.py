import copy
import math

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.constants import WARD_MAP, GOALS_LIST, GOAL_GROUPS
from social_dilemmas.envs.nurse_agent import NurseAgent

NON_URGENT_GOALS_MAX_REWARD = 30
URGENT_GOAL_MAX_REWARD = 1000
DECAY_CONSTANT = -0.2


class NurseEnv(MapEnv):

    def __init__(self, ascii_map=WARD_MAP, num_agents=1, render=False):
        # self.reward = {goal: 1 for goal in DEFAULT_GOALS}
        # self.reward.update(reward)
        super().__init__(ascii_map, num_agents, render)
        self.pos_dict = {i: [] for i in GOALS_LIST}  # stores all possible coordinates of each goal
        self.goal_groups = GOAL_GROUPS
        self.respawn_prob = {i: 0.001 for i in GOALS_LIST}
        # make a dict of the potential goal (e.g., B, S, T) spawn points
        print('base_map shape', self.base_map.shape[0], self.base_map.shape[1])
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] in self.pos_dict:
                    self.pos_dict[self.base_map[row, col]].append([row, col])
        print('pos_dict', self.pos_dict)
        self.goal_scores_dict = {}
        self.timestep = 0
        self.goals_relative_timestep = {}
        # self.prev_goals_list = []
        # self.curr_goals_list = []

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def custom_map_update(self):
        spawn_points = self.spawn_ward_goal_v2()
        self.update_map(spawn_points)
        # print('spawn points', spawn_points)
        self.update_goal_scores(spawn_points=spawn_points, initial=False)

    def initial_map_update(self):
        spawn_points = self.spawn_initial_goals()  # spawn initial goals is only at initialization of goals.
        self.update_map(spawn_points)
        self.update_goal_scores(initial=True)

    def update_goal_scores(self, spawn_points=[], initial=True):
        if initial:
            self.goal_scores_dict = {i: NON_URGENT_GOALS_MAX_REWARD for i in GOALS_LIST}
            # self.goal_scores_dict['S'] = URGENT_GOAL_MAX_REWARD
            self.goals_relative_timestep = {i: 0 for i in GOALS_LIST}
        else:
            if spawn_points:  # new goal appeared. look for 'S' the urgent goal.
                for spawn_point in spawn_points:
                    if spawn_point[2] == 'S':
                        self.goal_scores_dict['S'] = URGENT_GOAL_MAX_REWARD
                        self.goals_relative_timestep['S'] = 0
                        break
            else:
                for i in GOALS_LIST:
                    if i == 'S':  # decay goal reward (doesnt matter if it exists or not, it will be handled in the computation of group reward
                        self.goals_relative_timestep[i] += 1
                        self.goal_scores_dict[i] = URGENT_GOAL_MAX_REWARD * math.exp(
                            DECAY_CONSTANT * self.goals_relative_timestep[i])
                    else:
                        self.goal_scores_dict[i] = NON_URGENT_GOALS_MAX_REWARD
            # print('goals scores',self.goal_scores_dict)
            # print('self.goals_relative_timestep S', self.goals_relative_timestep['S'])

    # def get_available_goals_list(self):
    #     agents = list(self.agents.values())
    #     available_subgoals = agents[0].get_available_subgoals(self.pos_dict)
    #     return available_subgoals

    def setup_agents(self):
        """
        Creates a list of agents where each agent is a NurseAgent object.
        Each NurseAgent is initialized by urgency and reward_dict:
            - urgency is initialized in self.urgency
            - reward_dict contains for each goal a sampled reward value (possible reward values is the index of REWARD_PRIOR)
        :return:
        None (objects are stored in self.agents)
        """
        map_with_agents = self.get_map_with_agents()
        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            # spawn_point = self.spawn_point()
            spawn_point = np.array([3, 7])
            print('agent spawn point', spawn_point)
            rotation = self.spawn_rotation()
            # reward_dict = {}
            # for goal in DEFAULT_GOALS:
            #     goal_reward_dist = REWARD_PRIOR[goal]
            #     goal_reward_tensor = torch.tensor(goal_reward_dist)     # convert to tensor
            #     cat_dist = dist.Categorical(goal_reward_tensor)         # convert to dist object
            #     sample = int(pyro.sample("reward", cat_dist))           # sample from the distribution, sample will be the index number of reward list
            #     reward_dict[goal] = sample                              # add to dict in the form: {goal: sample}
            # print(agent_id, 'REWARD', reward_dict)

            # agent = NurseAgent(agent_id, spawn_point, rotation, map_with_agents, self.urgency, reward_dict)
            agent = NurseAgent(agent_id, spawn_point, rotation, map_with_agents)
            self.agents[agent_id] = agent

    def spawn_ward_goal(self):
        """
        Creates a list of randomly selected spawn points of tasks depending on the spawn probability
        A goal can be any of 'B', 'S', or 'T'
        :return:
        spawn_points: list of spawn points where each element is tuple(row, col, goal)
        """
        spawn_points = []
        for goal in self.pos_dict:
            for i in self.pos_dict[goal]:
                row, col = i  # get a coordinate of a possible goal location

                # condition: if world_map label ('@', 'X') is not a goal ['B', 'S', 'T']
                # and if current coordinate is not the agent position
                # then get a random number
                # the random number is compared against the spawn probability.
                # if it's less than the spawn probability, then create that spawn point.
                if self.world_map[row, col] not in self.pos_dict and [row, col] not in self.agent_pos:
                    rand_num = np.random.rand(1)[0]
                    if rand_num < self.respawn_prob[goal]:
                        spawn_points.append((row, col, goal))

        return spawn_points

    def spawn_ward_goal_v2(self):
        """
        :return:
        spawn_points: list of spawn points where each element is tuple(row, col, goal)
        """
        spawn_points = []
        '''
        for goal in self.pos_dict:
            #print(goal)
            for i in self.pos_dict[goal]:
                row, col = i               # get a coordinate of a possible goal location              
                # condition: if world_map label ('@', 'X') is not a goal ['B', 'S', 'T']
                # and if current coordinate is not the agent position
                # then get a random number
                # the random number is compared against the spawn probability.
                # if it's less than the spawn probability, then create that spawn point.
                if self.world_map[row, col] not in self.pos_dict and [row, col] not in self.agent_pos:
                    #print('getting random number')
                    rand_num = np.random.rand(1)[0]
                    #rand_num = 10
                    if rand_num < self.respawn_prob[goal]:
                        spawn_points.append((row, col, goal))

        '''
        for goal_group, goal_group_attributes in self.goal_groups.items():
            for goal, goal_attributes in goal_group_attributes['goals'].items():
                row, col = self.pos_dict[goal][0]
                # print(goal_attributes['obtainable'])

                if goal_attributes['obtainable'] == True and goal_attributes['taken'] == False:
                    # print('goal attainable')
                    if self.world_map[row, col] not in self.pos_dict and [row, col] not in self.agent_pos:
                        # print('getting random number')
                        rand_num = np.random.rand(1)[0]
                        # rand_num = 10
                        if rand_num < self.respawn_prob[goal]:
                            spawn_points.append((row, col, goal))
                elif goal_attributes['obtainable'] == False and goal_attributes['taken'] == False:
                    spawn_points.append((row, col, goal))

        return spawn_points

    def spawn_initial_goals(self):
        spawn_points = []
        for goal in self.pos_dict:
            for i in self.pos_dict[goal]:
                row, col = i  # get a coordinate of a possible goal location
                if self.world_map[row, col] not in self.pos_dict and [row, col] not in self.agent_pos:
                    spawn_points.append((row, col, goal))

        return spawn_points
