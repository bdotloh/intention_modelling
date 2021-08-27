import copy
import math

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.constants import WARD_MAP, WARD_MAP_2, GOALS_LIST, WARD_MAP_V2_1
from social_dilemmas.envs.nurse_agent import NurseAgent

NON_URGENT_GOALS_MAX_REWARD = 30
URGENT_GOAL_MAX_REWARD = 1000
DECAY_CONSTANT = -0.05


class NurseEnv(MapEnv):

    def __init__(self, ascii_map=WARD_MAP_V2_1, num_agents=1, render=False):
        # self.reward = {goal: 1 for goal in DEFAULT_GOALS}
        # self.reward.update(reward)
        super().__init__(ascii_map, num_agents, render)

        self.goals_dict = GOALS_LIST  # store goals' attribute (coordinates, requirements, urgency)

        # print('BASE_MAP',self.base_map)

        # add goals' location to goal dict
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] in self.goals_dict:
                    self.goals_dict[self.base_map[row, col]]['location'] = (row, col)

        # print('goals_dict', self.goals_dict)

        self.respawn_prob = {i: 0.1 for i in GOALS_LIST}  # make dict of potential goal spawn points
        self.goal_scores_dict = {i: 0 for i in self.goals_dict.keys()}
        self.urgent_goals_relative_timestep = {}
        self.timestep = 0
        self.new_goal_appear = None

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def custom_map_update(self):
        spawn_points = self.spawn_ward_goal()
        if spawn_points:
            self.new_goal_appear = True
        else:
            self.new_goal_appear = False
        self.update_map(spawn_points)
        # print('spawn points', spawn_points)
        self.update_goal_scores(spawn_points=spawn_points)

    def initial_map_update(self):
        spawn_points = self.spawn_initial_goals()  # spawn initial goals is only at initialization of goals.
        self.update_map(spawn_points)
        self.update_goal_scores(spawn_points)

    def update_goal_scores(self, spawn_points=[]):
        agent = list(self.agents.values())[0]
        if spawn_points:
            for spawn_point in spawn_points:
                goal = spawn_point[2]
                if self.goals_dict[goal]['requires']:
                    self.goal_scores_dict[goal] = 0

                elif self.goals_dict[goal]['urgency']:
                    self.goal_scores_dict[goal] = URGENT_GOAL_MAX_REWARD

                else:
                    self.goal_scores_dict[goal] = NON_URGENT_GOALS_MAX_REWARD

            self.urgent_goals_relative_timestep = {goal: 0 for goal, attr in self.goals_dict.items() if attr['urgency']}

        for goal, attr in self.goals_dict.items():
            if attr['urgency']:
                self.goal_scores_dict[goal] = self.goal_scores_dict[goal] * math.exp(
                    DECAY_CONSTANT * self.urgent_goals_relative_timestep[goal])
                self.urgent_goals_relative_timestep[goal] += 1

            if attr['requires']:
                if all(goal in agent.possessions for goal in self.goals_dict[goal]['requires']):
                    self.goal_scores_dict['c'] = NON_URGENT_GOALS_MAX_REWARD

    def setup_agents(self):
        """
        Creates a list of agents where each agent is a NurseAgent object.
        Each NurseAgent is initialized by urgency and reward_dict:
            - urgency is initialized in self.urgency
            - reward_dict contains for each goal a sampled reward value (possible reward values is the index of REWARD_PRIOR)
        :return:
        None (objects are stored in self.agents)
        """

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            #spawn_point = np.array([7, 8])
            print('agent spawn point', spawn_point)
            rotation = self.spawn_rotation()
            agent = NurseAgent(agent_id, spawn_point, rotation, self.base_map,
                               state_beliefs=copy.deepcopy(self.goals_dict))
            self.agents[agent_id] = agent

    def spawn_ward_goal(self):
        """
        :return:
        spawn_points: list of spawn points where each element is tuple(row, col, goal)
        """
        spawn_points = []

        for goal, attributes in self.goals_dict.items():
            if self.timestep == 11:   #11
                if goal == 'd' or goal == 'e' or goal == 'f':
                    row, col = attributes['location']
                    if self.world_map[row, col] not in self.goals_dict and [row, col] not in self.agent_pos:
                        spawn_points.append((row,col,goal))

            if self.timestep == 2 or self.timestep == 11:  # 2, 11, 19
                if goal == 'S':
                    row, col = attributes['location']
                    if self.world_map[row, col] not in self.goals_dict and [row, col] not in self.agent_pos:
                        spawn_points.append((row,col,goal))

            if self.timestep == 2 or self.timestep == 8 or self.timestep == 10 or self.timestep == 15 or self.timestep ==18:   #2, 8, 10, 15, 18
                if goal == 'T':
                    row, col = attributes['location']
                    if self.world_map[row, col] not in self.goals_dict and [row, col] not in self.agent_pos:
                        spawn_points.append((row,col,goal))

        return spawn_points

    def spawn_initial_goals(self):
        spawn_points = []
        for goal, attr in self.goals_dict.items():
            if goal == 'a' or goal == 'b' or goal == 'c':
                row, col = attr['location']  # get xy of a possible goal location
                if self.world_map[row, col] not in self.goals_dict and [row, col] not in self.agent_pos:
                    spawn_points.append((row, col, goal))
        return spawn_points
