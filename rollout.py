"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import utility_funcs
import numpy as np
import os
import sys
import shutil
#import tensorflow as tf
from social_dilemmas.envs.nurse_env import NurseEnv
from social_dilemmas.observer import Observer

'''
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'vid_path', os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')),
    'Path to directory where videos are saved.')
tf.app.flags.DEFINE_string(
    'env', 'cleanup',
    'Name of the environment to rollout. Can be cleanup or harvest.')
tf.app.flags.DEFINE_string(
    'render_type', 'pretty',
    'Can be pretty or fast. Implications obvious.')
tf.app.flags.DEFINE_integer(
    'fps', 8,
    'Number of frames per second.')
'''


class Controller(object):

    def __init__(self, env_name='nurse'):
        self.env_name = env_name
        if env_name == 'nurse':
            print('Initializing Nurse environment')
            self.env = NurseEnv(num_agents=1, render=True)
        else:
            print('Error! Not a valid environment type')
            return

        self.env.reset()

        # TODO: initialize agents here

    def rollout(self, horizon=None, save_path=None):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
            save_path: If provided, will save each frame to disk at this
                location.
        """

        rewards = 0
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

        observer = Observer(list(self.env.agents.values())[0].grid.copy())

        for hor in range(horizon):
            print('GOALSCORES {}'.format(self.env.goal_scores_dict))
            agents = list(self.env.agents.values())   #list of agent objects
            observer.update_grid(agents[0].grid)      # update observer's env from agent's env (both fully observable)

            depth = 2000
            action_list = []
            print('--------timestep %s--------' % hor)

            for j in range(self.env.num_agents):
                act = agents[j].policy(depth, self.env.goal_scores_dict, self.env.goals_dict, self.env.new_goal_appear)
                action_list.append(act)

            obs, rew, dones, info, = self.env.step({'agent-%d' % k: action_list[k] for k in range(len(agents))})
            action_list.append(hor)    # very important for pyro to know that this observation is new. else if a

            sys.stdout.flush()

            if save_path is not None:
                self.env.render(filename=save_path + 'frame' + str(hor).zfill(6) + '.png')

            rgb_arr = self.env.map_to_colors()
            full_obs[hor] = rgb_arr.astype(np.uint8)
            rewards += rew['agent-0']
            # observations.append(obs['agent-0'])
            # rewards.append(rew['agent-0'])

        [print(map_with_agents) for map_with_agents in self.env.maps]
        print(rewards)

        return rewards, observations, full_obs

    def render_rollout(self, horizon=400, path=None,
                       render_type='pretty', fps=8):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out.
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
            fps: Integer frames per second.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
        video_name = self.env_name + '_trajectory'

        if render_type == 'pretty':
            image_path = os.path.join(path, 'frames/')
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            print('image_path', image_path)

            rewards, observations, full_obs = self.rollout(
                horizon=horizon, save_path=image_path)
            utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
                                                    video_name=video_name)

            # Clean up images
            # shutil.rmtree(image_path)
        else:
            rewards, observations, full_obs = self.rollout(horizon=horizon)
            utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=fps,
                                                   video_name=video_name)


def main(horizon = 50):
    # c = Controller(env_name=FLAGS.env)
    c = Controller(env_name='nurse')
    c.rollout(horizon)


if __name__ == '__main__':
    main()
