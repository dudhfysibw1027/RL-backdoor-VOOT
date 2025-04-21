import os
import time

from trajectory_representation.operator import Operator
import pickle
import numpy as np
import os
print(os.getcwd())
from problem_environments.LSTM_policy import LSTMPolicy
import gym
import gym_compete
import torch


class MultiAgentEnvTorchMitigation:
    def __init__(self, env_name='run-to-goal-humans-v0',
                 model_name="saved_models/human-to-go/trojan_model_128.h5", seed=0):
        # This is for multiagent environment
        # such as run-to-goal-ant, run-to-goal-humanoid
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.robot = None
        self.objects_currently_not_in_goal = []
        self.infeasible_reward = -2000
        self.problem_idx = 0
        self.name = 'multiagent'
        self.reward_function = None
        self.feasible_reward = None
        self.done_and_not_found = False
        now_win = os.getcwd().split('\\')
        now_lin = os.getcwd().split('/')
        print('now', now_lin[-1], now_win[-1])
        if now_win[-1] == 'test_scripts' or now_lin[-1] == 'test_scripts':
            model_name = model_name
        else:
            model_name = 'test_scripts/' + model_name
        self.oppo_model = torch.load(model_name).to('cuda')
        self.dim_x = self.env.action_space.spaces[0].shape[0]
        self.state_dim = self.env.observation_space.spaces[0].shape[0]
        self.action_dim = self.env.action_space.spaces[0].shape[0]
        self.seed = seed
        self.env.seed(self.seed)
        self.curr_state = self.env.reset()
        self.found_trigger = False
        self.feasible_action_value_threshold = -1000
        self.ant_anomaly_threshold = 48.083856548755605
        self.ant_anomaly_threshold_60 = 48.083856548755605
        self.ant_anomaly_threshold_100 = 129.3819963670962
        if 'test_scripts' not in os.getcwd():
            self.ant_anomaly_threshold_array = np.load('test_scripts/parameters/thresholds_0_to_100.npy')
        else:
            self.ant_anomaly_threshold_array = np.load('parameters/thresholds_0_to_100.npy')
        self.observing_phase_m = 50
        self.len_lstm_policy_input = 10
        if 'human' in env_name:
            if now_win[-1] == 'test_scripts' or now_lin == 'test_scripts':
                self.ob_mean = np.load(
                    "../test_scripts/parameters/human-to-go/obrs_mean.npy")
                self.ob_std = np.load(
                    "../test_scripts/parameters/human-to-go/obrs_std.npy")
            else:
                self.ob_mean = np.load(
                    "test_scripts/parameters/human-to-go/obrs_mean.npy")
                self.ob_std = np.load(
                    "test_scripts/parameters/human-to-go/obrs_std.npy")
        elif "ant" in env_name:
            if now_win[-1] == 'test_scripts' or now_lin == 'test_scripts':
                self.ob_mean = np.load(
                    "../test_scripts/parameters/ants_to_go/obrs_mean.npy")
                self.ob_std = np.load(
                    "../test_scripts/parameters/ants_to_go/obrs_std.npy")
            else:
                self.ob_mean = np.load(
                    "test_scripts/parameters/ants_to_go/obrs_mean.npy")
                self.ob_std = np.load(
                    "test_scripts/parameters/ants_to_go/obrs_std.npy")
        else:
            self.ob_mean = np.load(
                "test_scripts/parameters/human-to-go/obrs_mean.npy")
            # "parameters/ants_to_go/obrs_mean.npy")
            self.ob_std = np.load(
                "test_scripts/parameters/human-to-go/obrs_std.npy")
            # "parameters/ants_to_go/obrs_std.npy")
        self.policy0 = None
        self.policy1 = None
        self.done = False

    def set_policy(self, policy0, policy1):
        self.policy0 = policy0
        self.policy1 = policy1

    def reset_to_init_state(self, node, initial_state=None):
        # (original) todo reset to the original state. Do this by changing the reward function to the initial one.
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        print(f"reset to init state and seed={self.seed}")
        self.env.seed(self.seed)
        self.done = False
        if initial_state is None:
            self.curr_state = self.env.reset()
        else:
            ob1, ob2 = initial_state
            if 'human' in self.env_name:
                pos1 = np.array(ob1[0:24])  # .astype(self.observation_space.dtype)
                pos2 = np.array(ob2[0:24])  # .astype(self.observation_space.dtype)
                vel1 = np.array(ob1[24:47])  # .astype(self.observation_space.dtype)
                vel2 = np.array(ob2[24:47])  # .astype(self.observation_space.dtype)
            elif 'ant' in self.env_name:
                pos1 = np.array(ob1[0:15])  # .astype(self.observation_space.dtype)
                pos2 = np.array(ob2[0:15])  # .astype(self.observation_space.dtype)
                vel1 = np.array(ob1[15:29])  # .astype(self.observation_space.dtype)
                vel2 = np.array(ob2[15:29])  # .astype(self.observation_space.dtype)
            else:
                print('Not in multi-agent env')
                return -1
            qpos = np.concatenate((pos1, pos2), axis=0)
            qvel = np.concatenate([vel1, vel2])
            self.env.set_state(qpos, qvel)
            self.curr_state = initial_state
            node.set_node_state(initial_state)
        self.found_trigger = False
        self.done_and_not_found = False
        # TODO me: what does objects_currently_not_in_goal means?
        # objects_currently_not_in_goal means break
        self.objects_currently_not_in_goal = node.objects_not_in_goal

    def access_done_and_not_found(self):
        return self.done_and_not_found

    def set_env_seed(self, seed):
        self.seed = seed

    def set_node_state(self, node):
        state = node.state
        ob1, ob2 = state

        if 'human' in self.env_name:
            pos1 = np.array(ob1[0:24])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:24])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[24:47])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[24:47])  # .astype(self.observation_space.dtype)
        elif 'ant' in self.env_name:
            pos1 = np.array(ob1[0:15])  # .astype(self.observation_space.dtype)
            pos2 = np.array(ob2[0:15])  # .astype(self.observation_space.dtype)
            vel1 = np.array(ob1[15:29])  # .astype(self.observation_space.dtype)
            vel2 = np.array(ob2[15:29])  # .astype(self.observation_space.dtype)
        else:
            print('Not in multi-agent env')
        qpos = np.concatenate((pos1, pos2), axis=0)
        qvel = np.concatenate([vel1, vel2])
        # self.env.seed(self.seed)
        # self.env.reset()
        self.env.set_state(qpos, qvel)

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        action = operator_instance.continuous_parameters['action_parameters']
        state = node.state
        ob1, ob2 = state
        self.set_node_state(node)
        self.done = False
        # next_state_for_crate_node = state
        # if 'human' in self.env_name:
        #     pos1 = np.array(ob1[0:24])  # .astype(self.observation_space.dtype)
        #     pos2 = np.array(ob2[0:24])  # .astype(self.observation_space.dtype)
        #     vel1 = np.array(ob1[24:47])  # .astype(self.observation_space.dtype)
        #     vel2 = np.array(ob2[24:47])  # .astype(self.observation_space.dtype)
        # elif 'ant' in self.env_name:
        #     pos1 = np.array(ob1[0:15])  # .astype(self.observation_space.dtype)
        #     pos2 = np.array(ob2[0:15])  # .astype(self.observation_space.dtype)
        #     vel1 = np.array(ob1[15:29])  # .astype(self.observation_space.dtype)
        #     vel2 = np.array(ob2[15:29])  # .astype(self.observation_space.dtype)
        # qpos = np.concatenate((pos1, pos2), axis=0)
        # qvel = np.concatenate([vel1, vel2])
        #
        # self.env.set_state(qpos, qvel)

        a0, v0, s0, n0 = self.policy0.step(ob1, deterministic=True)
        next_state, r, d, _ = self.env.step(  # clipped_actions[0]
            ([a0, action]))
        # self.env.render()
        # time.sleep(0.1)
        one_step_reward = r[1]
        self.curr_state = next_state
        if d[0] or d[1]:
            self.done = True
        return one_step_reward  # , next_state_for_crate_node

    def apply_operator_instance(self, operator_instance, node):
        reward = self.apply_action_and_get_reward(operator_instance, True, node)
        print("Pure reward", reward)  # , next_state == -1)  # , "next_s", next_state[0].shape)

        # TODO me: what's feasible action value threshold?
        if reward < self.feasible_action_value_threshold:
            # reward = reward + self.infeasible_reward
            # # todo stop advancing if your reward is less than 0.3
            operator_instance.continuous_parameters['is_feasible'] = False
        else:
            # reward += self.feasible_reward
            operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def apply_action_and_get_reward_last(self, operator_instance, is_op_feasible, node):
        return 0

    def apply_operator_instance_last(self, operator_instance, node):
        reward = self.apply_action_and_get_reward_last(operator_instance, True, node)
        return reward

    def is_action_feasible(self, action, action_parameter=None):
        # TODO me: is this needed? i abandon it now
        # if action_parameter is None:
        #     reward, _ = self.apply_action_and_get_reward(action, True, None)
        # else:
        #     pass
        # reward = self.reward_function(action_parameter)
        # reward, _ = self.apply_action_and_get_reward(action, True, None)
        # return reward > self.feasible_action_value_threshold
        return True

    def is_goal_reached(self):
        # originally only return self.found_trigger
        return self.found_trigger

    def is_done(self):
        # originally only return self.found_trigger
        return self.done

    def check_trigger(self, trigger_action, seed, iteration):
        return False

    def get_applicable_op_skeleton(self, parent_action):
        op = Operator(operator_type='multiagent_' + str(self.dim_x),
                      discrete_parameters={},
                      continuous_parameters=None,
                      low_level_motion=None)
        return op

    def is_pick_time(self):
        return False

    def get_mean_std_dim(self):
        return self.ob_mean, self.ob_std, self.state_dim, self.action_dim

    def apply_action_and_get_reward_no_set_state(self, states, curr_state):
        ob1, ob2 = curr_state
        state_seq = [s[1] for s in states]
        state_seq_norm = [np.clip((s - self.ob_mean) / self.ob_std, -5.0, 5.0) for s in state_seq]
        state_seq_norm = np.array(state_seq_norm)
        state_seq_norm = np.reshape(state_seq_norm, (1, -1, self.state_dim))
        ob_tensor = torch.tensor(state_seq_norm).float().to('cuda')
        trojan_action = self.oppo_model.predict(ob_tensor).cpu()
        a0, v0, s0, n0 = self.policy0.step(ob1, deterministic=True)
        next_state, r, d, _ = self.env.step(
            ([a0, trojan_action[0]]))
        # self.env.render()
        # time.sleep(0.1)
        one_step_reward = r[1]
        self.curr_state = next_state
        return one_step_reward
