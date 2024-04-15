from trajectory_representation.operator import Operator

import pickle
import numpy as np

import gym
import gym_compete
import tensorflow.keras as keras
import tensorflow as tf


class MultiAgentEnv:
    def __init__(self, env_name='run-to-goal-humans-v0',
                 model_name="saved_models/human-to-go/trojan_model_128.h5", seed=0):
        # This is for multiagent environment
        # such as run-to-goal-ant, run-to-goal-humanoid
        self.env = gym.make(env_name)
        self.robot = None
        self.objects_currently_not_in_goal = []
        self.infeasible_reward = -2
        self.problem_idx = 0
        self.name = 'multiagent'
        self.reward_function = None
        self.feasible_reward = None
        self.oppo_model = keras.models.load_model(model_name)
        self.dim_x = self.env.action_space.spaces[0].shape[0]
        self.seed = seed
        self.env.seed(self.seed)
        self.curr_state = self.env.reset()

    def reset_to_init_state(self, node):
        # (original) todo reset to the original state. Do this by changing the reward function to the initial one.
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        self.env.seed(self.seed)
        self.curr_state = self.env.reset()
        # TODO me: what does objects_currently_not_in_goal means?
        self.objects_currently_not_in_goal = node.objects_not_in_goal

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        action = operator_instance.continuous_parameters['action_parameters']
        # todo make the action to change the next state's reward function
        #       how should I change it?
        #       one simple idea is to shift the shekel function around
        #
        state = node.state
        ob1, ob2 = state
        pos1 = np.array(ob1[0:24])  # .astype(self.observation_space.dtype)
        pos2 = np.array(ob2[0:24])  # .astype(self.observation_space.dtype)
        vel1 = np.array(ob1[24:47])  # .astype(self.observation_space.dtype)
        vel2 = np.array(ob2[24:47])  # .astype(self.observation_space.dtype)
        qpos = np.concatenate((pos1, pos2), axis=0)
        qvel = np.concatenate([vel1, vel2])
        self.env.set_state(qpos, qvel)

        # reward = self.reward_function(action)
        return reward

    def apply_operator_instance(self, operator_instance, node):
        reward = self.apply_action_and_get_reward(operator_instance, True, node)
        print("Pure reward", reward)
        if reward < self.feasible_action_value_threshold:
            reward = reward + self.infeasible_reward
            # todo stop advancing if your reward is less than 0.3
            operator_instance.continuous_parameters['is_feasible'] = False
        else:
            reward += self.feasible_reward
            operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def is_action_feasible(self, action, action_parameter=None):
        if action_parameter is None:
            reward = self.apply_action_and_get_reward(action, True, None)
        else:
            reward = self.reward_function(action_parameter)

        return reward > self.feasible_action_value_threshold

    def is_goal_reached(self):
        return False

    def get_applicable_op_skeleton(self, parent_action):
        op = Operator(operator_type='multiagent_' + str(self.dim_x),
                      discrete_parameters={},
                      continuous_parameters=None,
                      low_level_motion=None)
        return op

    def is_pick_time(self):
        return False


# class ShekelSynthetic(SyntheticEnv):
#     def __init__(self, problem_idx):
#         SyntheticEnv.__init__(self, problem_idx)
#         self.name = 'synthetic_shekel'
#         if problem_idx == 0:
#             self.dim_x = 3
#             self.feasible_action_value_threshold = 3.0
#         elif problem_idx == 1:
#             self.dim_x = 10
#             self.feasible_action_value_threshold = 2.0
#         elif problem_idx == 2:
#             self.dim_x = 20
#             self.feasible_action_value_threshold = 1.0
#         config = pickle.load(
#             open('./test_results/function_optimization/shekel/shekel_dim_' + str(self.dim_x) + '.pkl', 'r'))
#         A = config['A']
#         C = config['C']
#         self.reward_function = lambda sol: benchmarks.shekel(sol, A, C)[0]
#         self.feasible_reward = 1.0
#
#
# class RastriginSynthetic(SyntheticEnv):
#     def __init__(self, problem_idx, value_threshold):
#         SyntheticEnv.__init__(self, problem_idx)
#         self.name = 'synthetic_rastrigin'
#         if problem_idx == 0:
#             self.dim_x = 3
#             self.feasible_action_value_threshold = -10
#         elif problem_idx == 1:
#             self.dim_x = 10
#             self.feasible_action_value_threshold = value_threshold
#         elif problem_idx == 2:
#             self.dim_x = 20
#             self.feasible_action_value_threshold = -100
#
#         self.feasible_reward = 100
#         self.reward_function = lambda sol: -benchmarks.rastrigin(sol)[0]
#
#
# class GriewankSynthetic(SyntheticEnv):
#     def __init__(self, problem_idx):
#         SyntheticEnv.__init__(self, problem_idx)
#         self.name = 'synthetic_griewank'
#         if problem_idx == 0:
#             self.dim_x = 3
#             self.feasible_action_value_threshold = -2
#         elif problem_idx == 1:
#             self.dim_x = 10
#             self.feasible_action_value_threshold = -50
#         elif problem_idx == 2:
#             self.dim_x = 20
#             self.feasible_action_value_threshold = -50
#         self.feasible_reward = 100
#         self.reward_function = lambda sol: -benchmarks.griewank(sol)[0]
