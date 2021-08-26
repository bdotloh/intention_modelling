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
DECAY_CONSTANT = -0.2


class NurseEnv(MapEnv):

    def __init__(self, ascii_map=WARD_MAP_2, num_agents=1, render=False):
        # self.reward = {goal: 1 for goal in DEFAULT_GOALS}
        # self.reward.update(reward)
        super().__init__(ascii_map, num_agents, render)

        self.goals_dict = GOALS_LIST  # store goals' attribute (coordinates, requirements, urgency)

        # print('BASE_MAP',self.base_map)
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] in self.goals_dict:
                    self.goals_dict[self.base_map[row, col]]['location'] = (row, col)

        # print('goals_dict', self.goals_dict)

        self.respawn_prob = {i: 0.1 for i in GOALS_LIST}  # make dict of potential goal spawn points
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

    def custom_map_update(self, hor):
        spawn_points = self.spawn_ward_goal(hor)
        self.update_map(spawn_points)
        # print('spawn points', spawn_points)
        self.update_goal_scores(spawn_points=spawn_points, initial=False)

        # if hor == 1:
        #     self.goals_dict['b']['urgency'] = True

    def initial_map_update(self):
        spawn_points = self.spawn_initial_goals()  # spawn initial goals is only at initialization of goals.
        self.update_map(spawn_points)
        self.update_goal_scores(initial=True)

    def update_goal_scores(self, spawn_points=[], initial=True):
        agent = list(self.agents.values())[0]
        if initial:  # HARDCODE: c,d initialised as 0 score goals
            self.goal_scores_dict = {i: False for i in self.goals_dict.keys()}
            self.goal_scores_dict['a'] = NON_URGENT_GOALS_MAX_REWARD
            self.goal_scores_dict['b'] = NON_URGENT_GOALS_MAX_REWARD
            self.goal_scores_dict['c'] = 0
            self.goals_relative_timestep = {i: False for i in self.goals_dict.keys()}
        else:
            if spawn_points:  # new goal appeared. look for 'S' the urgent goal.
                for spawn_point in spawn_points:
                    goal = spawn_point[2]
                    if goal == 'S':
                        self.goal_scores_dict[goal] = URGENT_GOAL_MAX_REWARD
                        self.goals_relative_timestep[goal] = 0

                    elif goal == 'd':
                        self.goal_scores_dict[goal] = 0

                    else:
                        self.goal_scores_dict[goal] = NON_URGENT_GOALS_MAX_REWARD

            if self.goal_scores_dict['S']:
                self.goal_scores_dict['S'] = self.goal_scores_dict['S'] * math.exp(
                    DECAY_CONSTANT * self.goals_relative_timestep['S'])
                self.goals_relative_timestep['S'] += 1

            if all(goal in agent.possessions for goal in self.goals_dict['c']['requires']):
                self.goal_scores_dict['c'] = NON_URGENT_GOALS_MAX_REWARD

            # if all(goal in agent.possessions for goal in self.goals_dict['d']['requires']):
            #     self.goal_scores_dict['d'] = NON_URGENT_GOALS_MAX_REWARD

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
            spawn_point = np.array([7, 8])
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
            agent = NurseAgent(agent_id, spawn_point, rotation, map_with_agents,
                               state_beliefs=copy.deepcopy(self.goals_dict))
            self.agents[agent_id] = agent

    def spawn_ward_goal(self, hor):
        """
        :return:
        spawn_points: list of spawn points where each element is tuple(row, col, goal)
        """
        spawn_points = []

        if hor == 2 or hor == 11:
            print('spawning S')
            row, col = self.goals_dict['S']['location']
            spawn_points.append((row, col, 'S'))

        # for goal, attributes in self.goals_dict.items():
        #     if hor == 11:   #11
        #         if goal == 'd' or goal == 'e' or goal == 'f':
        #             row, col = attributes['location']
        #             spawn_points.append((row,col,goal))
        #
        #     if hor == 2 or hor == 11:  # 2, 11, 19
        #         if goal == 'S':
        #             row, col = attributes['location']
        #             spawn_points.append((row,col,goal))
        #
        #     elif hor == 2 or 8 or 10 or 15 or 18:   #2, 8, 10, 15, 18
        #         if goal == 'T':
        #             row, col = attributes['location']
        #             spawn_points.append((row,col,goal))

        # for goal, attributes in self.goals_dict.items():
        #     row, col = attributes['location']
        #     # get a coordinate of a possible goal location
        #     # condition: if world_map label ('@', 'X') is not a goal ['B', 'S', 'T']
        #     # and if current coordinate is not the agent position
        #     # then get a random number.
        #     # if random number < spawn probability, create that spawn point.
        #     if self.world_map[row, col] not in self.goals_dict and [row, col] not in self.agent_pos:
        #         # print('getting random number')
        #         rand_num = np.random.rand(1)[0]
        #         if rand_num < self.respawn_prob[goal]:
        #             spawn_points.append((row, col, goal))
        return spawn_points

    def spawn_initial_goals(self):
        spawn_points = []
        for goal, beliefs in self.goals_dict.items():
            if goal == 'a' or goal == 'b' or goal == 'c':
                row, col = beliefs['location']  # get xy of a possible goal location
                if self.world_map[row, col] not in self.goals_dict and [row, col] not in self.agent_pos:
                    spawn_points.append((row, col, goal))
        return spawn_points
