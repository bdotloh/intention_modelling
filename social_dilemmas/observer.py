import torch
import numpy as np
import pyro
import pyro.distributions as dist

from social_dilemmas.search_inference import factor, HashingMarginal, memoize, Search
from social_dilemmas.envs.nurse_agent import NurseAgent

REWARD_LIST = [0, 1, 2]
OBSERVER_REWARD_PRIOR = {0: [1/3, 1/3, 1/3],
                        1: [1/3, 1/3, 1/3],
                        2: [1/3, 1/3, 1/3]}
URGENCY_DICT = {0: "B",
                1: "S",
                2: "T"}


def Marginal(fn):  # takes a function as in input. returns all the possible enumerations (tuple of urgency, reward, action in this case)
    return memoize(lambda *args: HashingMarginal(Search(fn).run(*args)))

class Observer():
    def __init__(self, grid, explorer_reward= None):
        self.grid = grid                                    # initialize grid = looks like the map in constants
        self.agent_urgency = {'B': True, 'S': False, 'T': False}
        self.rew_prior = {}                                 # distribution over 27 possibilities
        self.urgency_prior = [1., 0, 0]
        self.agent_no = 0
        self.reward_dict = {}                           # {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 1, 0], 4: [0, 1, 1], 5: [0, 1, 2], 6: [0, 2, 0], 7: [0, 2, 1], 8: [0, 2, 2], 9: [1, 0, 0], 10: [1, 0, 1], 11: [1, 0, 2], 12: [1, 1, 0], 13: [1, 1, 1], 14: [1, 1, 2], 15: [1, 2, 0], 16: [1, 2, 1], 17: [1, 2, 2], 18: [2, 0, 0], 19: [2, 0, 1], 20: [2, 0, 2], 21: [2, 1, 0], 22: [2, 1, 1], 23: [2, 1, 2], 24: [2, 2, 0], 25: [2, 2, 1], 26: [2, 2, 2]}
        self.update_reward_dict()                       # {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 1, 0], 4: [0, 1, 1], 5: [0, 1, 2], 6: [0, 2, 0], 7: [0, 2, 1], 8: [0, 2, 2], 9: [1, 0, 0], 10: [1, 0, 1], 11: [1, 0, 2], 12: [1, 1, 0], 13: [1, 1, 1], 14: [1, 1, 2], 15: [1, 2, 0], 16: [1, 2, 1], 17: [1, 2, 2], 18: [2, 0, 0], 19: [2, 0, 1], 20: [2, 0, 2], 21: [2, 1, 0], 22: [2, 1, 1], 23: [2, 1, 2], 24: [2, 2, 0], 25: [2, 2, 1], 26: [2, 2, 2]}
        self.probability_dict = {}
        self.get_agent_locs()                           # update self.agent_no
        self.REWARD_PRIOR = {}
        for ag in range(self.agent_no):
            self.REWARD_PRIOR["agent-{}".format(ag)] = OBSERVER_REWARD_PRIOR

        # TODO: If explorer is needed, uncomment below code
        # take in explorer reward sample, then update the inferred overarching distribution
        # self.REWARD_PRIOR = {}
        # for ag in range(self.agent_no):
        #     self.REWARD_PRIOR["agent-{}".format(ag)] = explorer_reward

    def update_reward_dict(self):
        """
        Creates a reward dictionary of values: possible combinations of reward values
        :return: None
        Updates self.reward_dict
        """
        #create reward_dict for easy interpretation of categorical distribution result
        length = len(REWARD_LIST)
        for a in range(length):
            for b in range(length):
                for c in range(length):
                    self.reward_dict[(length**2)*a+length*b+c] = [REWARD_LIST[a], REWARD_LIST[b], REWARD_LIST[c]]

    def get_agent_locs(self):
        """
        Gets locations of agents from one observation from the grid
        Agents are labeled by numbers '0-9'
        This also updates the number of agents observed and is stored in agent_no
        :return:
        agent_locs: dictionary of key=agent_id, value: agent location list [x, y]
        """
        agent_locs = {}
        agent_no = 0
        for row_elem in range(self.grid.shape[0]):
            for col_elem in range(self.grid.shape[1]):
                if self.grid[row_elem][col_elem] in "0123456789":
                    agent_locs[int(self.grid[row_elem][col_elem]) - 1] = ([row_elem, col_elem])
                    agent_no += 1
        self.agent_no = agent_no
        return agent_locs

    def update_grid(self, grid):
        self.grid = grid

    def observation(self, action):
        """

        :param action: list of actions actually performed by the each agent
        :return:
        """
        marginal = self.model(action)            # when model is called it gets updated, then marginal class the holds the enumeration of (urgency, reward, action)
        support = marginal.enumerate_support()   # note: total combinations is 27x3 = 81, enumerations [(0, 0, 4), (0, 1, 4), (0, 2, 4), (0, 3, 4), (0, 4, 4), (0, 5, 4), (0, 6, 4), (0, 7, 4), (0, 8, 4), (0, 9, 4), (0, 10, 4), (0, 11, 4), (0, 12, 4), (0, 13, 4), (0, 14, 4), (0, 15, 4), (0, 16, 4), (0, 17, 4), (0, 18, 4), (0, 19, 4), (0, 20, 4), (0, 21, 4), (0, 22, 4), (0, 23, 4), (0, 24, 4), (0, 25, 4), (0, 26, 4), (1, 0, 4), (1, 1, 4), (1, 2, 4), (1, 3, 4), (1, 4, 4), (1, 5, 4), (1, 6, 4), (1, 7, 4), (1, 8, 4), (1, 9, 4), (1, 10, 4), (1, 11, 4), (1, 12, 4), (1, 13, 4), (1, 14, 4), (1, 15, 4), (1, 16, 4), (1, 17, 4), (1, 18, 4), (1, 19, 4), (1, 20, 4), (1, 21, 4), (1, 22, 4), (1, 23, 4), (1, 24, 4), (1, 25, 4), (1, 26, 4), (2, 0, 4), (2, 1, 4), (2, 2, 4), (2, 3, 4), (2, 4, 4), (2, 5, 4), (2, 6, 4), (2, 7, 4), (2, 8, 4), (2, 9, 4), (2, 10, 4), (2, 11, 4), (2, 12, 4), (2, 13, 4), (2, 14, 4), (2, 15, 4), (2, 16, 4), (2, 17, 4), (2, 18, 4), (2, 19, 4), (2, 20, 4), (2, 21, 4), (2, 22, 4), (2, 23, 4), (2, 24, 4), (2, 25, 4), (2, 26, 4)]
        data = [marginal.log_prob(s).exp().item() for s in support]  # get log probability for each enumeration
        self.probability_dict = {support[index]: data[index] for index in range(len(support))}   # put in dict  {(0, 0, 3): 1.4210866573663265e-14, (0, 1, 3): 1.4210866573663265e-14, (0, 2, 3): 1.4210866573663265e-14, (0, 3, 3): 1.4210866573663265e-14, (0, 4, 3): 1.4210866573663265e-14, (0, 5, 3): 1.4210866573663265e-14, (0, 6, 3): 1.4210866573663265e-14, (0, 7, 3): 1.4210866573663265e-14, (0, 8, 3): 1.4210866573663265e-14, (0, 9, 3): 0.0555555522441864, (0, 10, 3): 0.0555555522441864, (0, 11, 3): 0.0555555522441864, (0, 12, 3): 0.0555555522441864, (0, 13, 3): 0.0555555522441864, (0, 14, 3): 0.0555555522441864, (0, 15, 3): 0.0555555522441864, (0, 16, 3): 0.0555555522441864, (0, 17, 3): 0.0555555522441864, (0, 18, 3): 0.0555555522441864, (0, 19, 3): 0.0555555522441864, (0, 20, 3): 0.0555555522441864, (0, 21, 3): 0.0555555522441864, (0, 22, 3): 0.0555555522441864, (0, 23, 3): 0.0555555522441864, (0, 24, 3): 0.0555555522441864, (0, 25, 3): 0.0555555522441864, (0, 26, 3): 0.0555555522441864, (1, 0, 3): 1.694064884766642e-21, (1, 1, 3): 1.694064884766642e-21, (1, 2, 3): 1.694064884766642e-21, (1, 3, 3): 1.4210866573663265e-14, (1, 4, 3): 1.4210866573663265e-14, (1, 5, 3): 1.4210866573663265e-14, (1, 6, 3): 1.4210866573663265e-14, (1, 7, 3): 1.4210866573663265e-14, (1, 8, 3): 1.4210866573663265e-14, (1, 9, 3): 7.894944756033485e-16, (1, 10, 3): 7.894944756033485e-16, (1, 11, 3): 7.894944756033485e-16, (1, 12, 3): 6.62274413087971e-09, (1, 13, 3): 6.62274413087971e-09, (1, 14, 3): 6.62274413087971e-09, (1, 15, 3): 6.62274413087971e-09, (1, 16, 3): 6.62274413087971e-09, (1, 17, 3): 6.62274413087971e-09, (1, 18, 3): 7.894944756033485e-16, (1, 19, 3): 7.894944756033485e-16, (1, 20, 3): 7.894944756033485e-16, (1, 21, 3): 6.62274413087971e-09, (1, 22, 3): 6.62274413087971e-09, (1, 23, 3): 6.62274413087971e-09, (1, 24, 3): 6.62274413087971e-09, (1, 25, 3): 6.62274413087971e-09, (1, 26, 3): 6.62274413087971e-09, (2, 0, 3): 1.694064884766642e-21, (2, 1, 3): 1.694064884766642e-21, (2, 2, 3): 1.694064884766642e-21, (2, 3, 3): 1.694064884766642e-21, (2, 4, 3): 1.694064884766642e-21, (2, 5, 3): 1.694064884766642e-21, (2, 6, 3): 1.694064884766642e-21, (2, 7, 3): 1.694064884766642e-21, (2, 8, 3): 1.694064884766642e-21, (2, 9, 3): 7.894944756033485e-16, (2, 10, 3): 7.894944756033485e-16, (2, 11, 3): 7.894944756033485e-16, (2, 12, 3): 7.894944756033485e-16, (2, 13, 3): 7.894944756033485e-16, (2, 14, 3): 7.894944756033485e-16, (2, 15, 3): 7.894944756033485e-16, (2, 16, 3): 7.894944756033485e-16, (2, 17, 3): 7.894944756033485e-16, (2, 18, 3): 7.894944756033485e-16, (2, 19, 3): 7.894944756033485e-16, (2, 20, 3): 7.894944756033485e-16, (2, 21, 3): 7.894944756033485e-16, (2, 22, 3): 7.894944756033485e-16, (2, 23, 3): 7.894944756033485e-16, (2, 24, 3): 7.894944756033485e-16, (2, 25, 3): 7.894944756033485e-16, (2, 26, 3): 7.894944756033485e-16}

        #compute urgency, reward
        urgency = {}
        reward_prior_update = {}
        reward = {"agent-{}".format(index):[0*i for i in range(len(self.agent_urgency))] for index in range(self.agent_no)}
        for agent_no in range(self.agent_no):
            reward_prior_update["agent-{}".format(agent_no)]={i:[0*j for j in range(len(self.REWARD_PRIOR["agent-0"][0]))]
                                                              for i in range(len(self.REWARD_PRIOR["agent-0"]))}
        for key in self.probability_dict:
            #calculate urgency
            # add all the probabilities corresponding to a specific urgency i.e., add all probabilities of urgency = 0, store it in urgency
            urgency[key[0]] = urgency[key[0]] + self.probability_dict[key] if key[0] in urgency else self.probability_dict[key]

            #calculate reward   # translate the probability dictionary to more readable format and store it in reward
            for index in range(self.agent_no):
                reward_list = self.reward_dict[key[index+1]]
                # increment possibility for three reward values for each agent
                new_reward_list = [x * self.probability_dict[key] for x in reward_list]
                reward["agent-{}".format(index)] = [new_reward_list[i]+reward["agent-{}".format(index)][i] \
                                                    for i in range(len(self.agent_urgency))]
                reward_tuple = self.reward_dict[key[index+1]] #reward list
                for reward_no in range(len(reward_tuple)):
                    reward_prior_update["agent-{}".format(index)][reward_no][reward_tuple[reward_no]] += self.probability_dict[key]

        # print("URGENCY: ", urgency)
        # print("REWARD: ", reward)  # this reward seems like absolute value of reward.
        #update urgency prior
        for i in range(len(self.agent_urgency)):
            self.urgency_prior[i] = urgency[i]
        #update reward_prior
        self.REWARD_PRIOR = reward_prior_update.copy()
        # print('REWARD_PRIOR', self.REWARD_PRIOR)  # this reward_prior contains the probability of the reward.
        return urgency, reward


    @Marginal
    def model(self, data):
        """
        Defines model as action conditioned upon the urgency and rewards priors
        step1: sample urgency
        step2: map sampled urgency to self.agent_urgency e.g., if sampled = 0, then B=True, S=False, T=False
        step3: create agents using sampled urgency and rewards
        step4: for each agent:
            - get the optimal policy e.g., action = 3
            - create a list of probability for each possible action e.g., if action = 3, action_prob_list = [0, 0, 0, 1, 0, 0]
        step5: sample the model for an action, and update model using observation
        :param data: list of observed actions performed by each agent
        :return:
        combined_params: tuple of combined parameters (urgency, reward, action)
        """
        u_sample = self.sample_urgency()   # sample from a 0,1,2 urgency. and set agent_urgency.
        # if sampled: urgency = True, else: urgency = False
        for goal in self.agent_urgency:  # {'B': True, 'S': False, 'T': False}
            if goal == URGENCY_DICT[int(u_sample)]:
                self.agent_urgency[goal] = True
            else:
                self.agent_urgency[goal] = False

        agent_list, r_sample = self.agent()
        action_prob_list = []
        for ag in agent_list:
            deterministic_action = ag.policy(2)
            action_prob = torch.zeros(5)             # total number of actions is 5
            action_prob[deterministic_action] = 1.0  # create a prior for action which is deterministic. e.g., [0., 0., 1., 0., 0.]
            action_prob_list.append(action_prob)
        # print('action_prob_list', action_prob_list) # action_prob_list [tensor([0., 0., 1., 0., 0.]), tensor([0., 0., 1., 0., 0.]), tensor([0., 0., 1., 0., 0.])]

        act = {}
        # model is updated (because action is conditioned upon urgency and rewards, therefore urgency and reward models get updated in the backend).
        # it gets updated using the observation data.
        # act is a sample from the action which is a deterministic function of urgency and rewards.
        # act is sampled from the "old" model before the update. only after sampling, then model is updated with observation data.
        for j in range(self.agent_no):
            act["act%d"%j] = pyro.sample('action%d'%j, dist.Categorical(probs=action_prob_list[j]), obs=torch.tensor(data[j]))
            if int(act["act%d"%j]) != data[j]:
                print('sampled and obs not equal!', 'act', act["act%d"%j], 'obs', data[j])
        combined_params = tuple()
        combined_params += (u_sample.item(),)
        for ag in range(self.agent_no):
            combined_params += (r_sample["agent-{}".format(ag)].item(),)
        for k in range(self.agent_no):
            combined_params += (act["act{}".format(k)].item(), )
        return combined_params
        # return (u_sample.item(),) + tuple(rew["agent-{}".format(ag)].item() for ag in range(self.agent_no)) + \
        #        tuple(act["act{}".format(k)].item() for k in range(self.agent_no))

    def agent(self):
        """
        Creates agents:
        step1: sample a reward value from all possible reward combinations (1-27). this is r_sample
        step2: convert r_sample to a "reward" dictionary {'B': 2, 'S': 1, 'T': 0} for each agent
        step3: create nurse agent using agent_locs, grid, agent_urgency and reward (from step2)
        :return:
        agent list: each nurse agent is created using the sampled urgency and sampled reward
        r_sample: dict of agents sampled reward in the format: {agent-x: tensor(sampled reward))
        """
        agent_locs = self.get_agent_locs()
        agent_list = []
        r_sample = self.sample_reward()    # r_sample  = {'agent-0': tensor(21), 'agent-1': tensor(3)}
        for i in range(self.agent_no):
            index = 0
            reward = {}
            for goal_id in self.agent_urgency:
                reward[goal_id] = self.reward_dict[int(r_sample['agent-' + str(i)])][index]   # convert sampled reward to a reward dict.
                index += 1
            # print('agent reward', reward)   # {'B': 1, 'S': 0, 'T': 1}
            # print('agent urgency', self.agent_urgency) # {'B': True, 'S': False, 'T': False}
            ag = NurseAgent("agent%d" % i, agent_locs[i], 'UP', self.grid, self.agent_urgency, reward)
            agent_list.append(ag)
        return agent_list, r_sample

    def sample_urgency(self):
        """
        step1: initialize a prob_tensor to a zeros array of length same as agent_urgency
        step2: get total of urgency_prior and normal urgency prior
        step3: store normalized urgency_prior to prob_tensor
        step4: create a categorical distribution using prob_tensor
        :return:
        urgency_prior: sampled from categorical distribution (based on length of agent_urgency)

        """
        prob_tensor = torch.zeros(len(self.agent_urgency))   #  {'B': True, 'S': False, 'T': False}
        total = sum(self.urgency_prior)                      # [1., 0, 0]
        temp_list = [i / total for i in self.urgency_prior]  # normalize  [1., 0, 0]
        for i in range(len(self.agent_urgency)):   # convert temp_list into a tensor (prob_tensor)
            prob_tensor[i] = temp_list[i]
        urgency_prior = pyro.sample("urgency", dist.Categorical(prob_tensor))
        return urgency_prior

    def sample_reward(self):
        """
        step1: Create a zeros tensor of length same as total possibilities of reward: self.rew_prior
        step2: Use REWARD_PRIOR to update self.rew_prior
            REWARD_PRIOR is a dictionary of reward_possibity: reward_probability for each reward value
            e.g., {0: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], 1: [...]

        :return:
        """
        reward_prior = {}
        #set self.reward_prior across 27 possibility
        for k in range(self.agent_no):
            self.rew_prior['agent-%d'%k] = torch.zeros(len(REWARD_LIST)**len(URGENCY_DICT))

        #rew_prior according to REWARD_PRIOR  # TODO: Ask LUHUI what does rew_prior represent?
        for agent_no in range(self.agent_no):
            for rew_no in range(len(self.rew_prior['agent-%d'%agent_no])):
                prior = 1
                for goal_id in URGENCY_DICT:
                    rew = self.reward_dict[rew_no][goal_id]
                    prior = prior * self.REWARD_PRIOR["agent-{}".format(agent_no)][goal_id][rew]
                self.rew_prior['agent-%d'%agent_no][rew_no] = prior
        for j in range(self.agent_no):
            reward_prior["agent-" + str(j)] = pyro.sample("reward_agent" + str(j), \
                                                          dist.Categorical(self.rew_prior["agent-"+str(j)]))

        return reward_prior
