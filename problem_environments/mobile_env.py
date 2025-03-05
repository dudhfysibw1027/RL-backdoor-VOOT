import copy
import os
import pickle
from collections import defaultdict
from itertools import product
from time import sleep
from typing import Dict

import torch

from trajectory_representation.operator import Operator
from mobile_env.core.base import MComCore
from mobile_env.core.entities import BaseStation, UserEquipment

from problem_environments.LSTM_policy import LSTMPolicyMultiDiscrete
from mobile_env.handlers.central import MComCentralHandler
import numpy as np


class MobileEnv:
    def __init__(self, env=None, model_name="saved_models/mobile_env/Trojan_2.pth", seed=0, len_lstm_policy_input=8,
                 env_name="mobile_env", actual_depth_limit=8, dimension_modification=None):
        if dimension_modification is None:
            dimension_modification = [3]
        self.env_name = env_name
        if env is None:
            self.env = CustomEnv(config={"handler": CustomHandler}, render_mode='human')
        else:
            self.env = env
        self.objects_currently_not_in_goal = []
        self.infeasible_reward = -1
        self.done_and_not_found = False
        self.name = 'mobile_env'
        self.reward_function = None
        self.feasible_reward = None
        self.robot = None
        now_win = os.getcwd().split('\\')
        now_lin = os.getcwd().split('/')
        print('now', now_lin[-1], now_win[-1])
        if now_win[-1] == 'test_scripts' or now_lin[-1] == 'test_scripts':
            model_name = model_name
        if 'test_scripts' in os.getcwd():
            model_name = model_name
        else:
            model_name = os.path.join('test_scripts', model_name)
        self.seed = 0
        self.env.seed = 0
        self.curr_state, self.curr_info = self.env.reset()
        self.curr_state_detail = self.env.get_state()
        self.found_trigger = False
        self.len_lstm_policy_input = len_lstm_policy_input
        print(os.getcwd(), model_name)
        self.trojan_model = torch.load(model_name).to('cuda')
        self.feasible_action_value_threshold = -1
        self.state_dim = self.env.observation_space.shape[0]
        self.mobile_threshold = 48.46003484

        self.ue_connection_count = {1: 0, 2: 0, 3: 0}
        self.ue_connection_count_trigger = {1: 0, 2: 0, 3: 0}
        self.ue_connection_count_after = {1: 0, 2: 0, 3: 0}
        self.ue_disconnection_count = {1: 0, 2: 0, 3: 0}

        self.init_ue_connection_count = {1: 0, 2: 0, 3: 0}

        self.all_ues = {1, 2, 3}
        self.disconnection = np.zeros((3, 2))
        self.disconnection_no_after = np.zeros((3, 2))
        # self.dim_x = len(dimension_modification)
        self.dim_x = 2
        # self.len_lstm_policy_input = len_lstm_policy_input
        self.actual_depth_limit = actual_depth_limit
        self.init_state_detail = self.curr_state_detail
        self.init_info = self.curr_info
        self.init_state = self.curr_state
        self.init_stable_connection = 0
        self.init_state_for_check = None
        self.init_info_for_check = None
        self.init_state_detail_for_check = None
        self.pure_model_name = model_name.split("/")[-1].split(".")[0]
        self.dimension_modification = dimension_modification

    def first_init_state(self, node):
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        print(f"first_init_state, seed={self.seed}")
        self.env.seed = self.seed
        obs, info = self.env.reset()
        self.init_state_for_check, self.init_info_for_check = obs, info
        self.init_state_detail_for_check = self.env.get_state()
        self.init_ue_connection_count = {1: 0, 2: 0, 3: 0}

        state_seq = []
        step = 0
        print("check conn")
        for _ in range(10):
            if 7 <= step < 10:
                connected_ues = set()
                for ue_list in info["connections"].values():
                    connected_ues.update(ue_list)
                for ue in connected_ues:
                    self.init_ue_connection_count[ue] += 1
            if step == 0:
                reset = True
            else:
                reset = False
            state_seq.append(obs)
            if len(state_seq) > self.len_lstm_policy_input:
                state_seq.pop(0)
            state_seq_np = np.reshape(state_seq, (1, -1, self.state_dim))
            ob_tensor = torch.tensor(state_seq_np).float().to('cuda')
            if step >= 8:
                print("obs_shape in first_init:", ob_tensor.shape)  # check already
            action_trojan = self.trojan_model.predict(ob_tensor, reset)  # if len_episode < 10:
            obs, reward, terminated, truncated, info = self.env.step(action_trojan)
            # self.env.render()
            step += 1
        node.state_sequence = state_seq
        self.found_trigger = False
        self.objects_currently_not_in_goal = node.objects_not_in_goal
        self.init_info = info
        self.init_state = obs
        self.init_state_detail = self.env.get_state()
        self.curr_state_detail = self.env.get_state()
        self.curr_state = obs
        self.curr_info = info

        node.state_detail = self.curr_state_detail
        print("init time:", node.state_detail["time"], "should be 10")
        node.state = self.curr_state
        self.init_stable_connection = 0
        for ue, count in self.init_ue_connection_count.items():
            # print("==", ue, count, "==")
            if count == 3:
                self.init_stable_connection += 1
        if self.init_stable_connection == 3:
            print("find stable", self.init_stable_connection)
            return True
        else:
            print("cannot find stable", self.init_stable_connection)
            return False

    def reset_to_init_state(self, node, initial_state=None):
        # (original) todo reset to the original state. Do this by changing the reward function to the initial one.
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        print(f"reset to init state and seed={self.seed}")
        self.env.seed = self.seed
        self.env.set_state(node.state_detail)
        # self.ue_connection_count = {1: 0, 2: 0, 3: 0}
        self.ue_connection_count = self.init_ue_connection_count
        self.ue_connection_count_trigger = {1: 0, 2: 0, 3: 0}
        self.ue_connection_count_after = {1: 0, 2: 0, 3: 0}
        self.ue_disconnection_count = {1: 0, 2: 0, 3: 0}
        self.disconnection = np.zeros((3, 2))
        self.disconnection_no_after = np.zeros((3, 2))

        self.found_trigger = False
        self.objects_currently_not_in_goal = node.objects_not_in_goal
        # self.curr_state_detail = self.init_state_detail
        # self.curr_state = self.init_state
        # self.curr_info = self.init_info
        self.curr_state_detail = self.env.get_state()
        self.curr_state = self.init_state
        self.curr_info = self.init_info
        # print("s_0_state==init_state", self.curr_state == self.init_state)
        # print("s_0_state_detail==init_state_detail", self.curr_state_detail == self.init_state_detail)
        # print(self.curr_state_detail['time'], self.init_state_detail['time'])
        # print(self.curr_state_detail)
        # print(self.init_state_detail)
        node.state_detail = self.curr_state_detail
        # print("init time:", node.state_detail["time"], "should be 10")
        node.state = self.curr_state
        # print("node.state_detail", node.state_detail)

    def set_env_seed(self, seed):
        self.seed = seed

    def access_done_and_not_found(self):
        # TODO: mobile-env done
        return self.done_and_not_found

    def set_node_state(self, node):
        state = node.state_detail
        self.env.set_state(state)

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        # print("operator_instance", operator_instance)
        action = operator_instance.continuous_parameters['action_parameters']
        state = node.state_detail
        # print("action:", action, ", time:", state["time"])
        # print("mcts_6: ", node.state_detail["time"])
        # print("apply time", state["time"])
        # print(state)
        self.env.set_state(state)

        state_seq = node.get_state_sequence()
        if len(state_seq) > self.len_lstm_policy_input:
            state_seq.pop(0)
        state_seq = np.reshape(state_seq, (1, -1, self.state_dim))
        # print("state_seq", state_seq.shape, state_seq.shape[1] - 1)
        if node.depth < self.actual_depth_limit:
            idx = 0
            if len(self.dimension_modification) == 1:
                state_seq[0][state_seq.shape[1] - 1][self.dimension_modification[0]] = action
            else:
                for n_dim in self.dimension_modification:
                    print("mobile_env_action", action, n_dim)
                    state_seq[0][state_seq.shape[1] - 1][n_dim] = action[idx]
                    idx += 1
        # print(state_seq)
        # print("Action", action)
        ob_tensor = torch.tensor(state_seq).float().to('cuda')
        # print("obs_shape in apply_action:", ob_tensor.shape)  # check ([1, 5, 18])
        allocator_action = self.trojan_model.predict(ob_tensor, reset=False)

        print("time", state["time"])
        next_state, r, term, trunc, info = self.env.step(allocator_action)
        # sleep(1)
        # self.env.render()
        one_step_reward = -r

        self.curr_state = next_state
        self.curr_state_detail = self.env.get_state()
        if 'mobile_env' in self.env_name:
            connected_ues = set()
            for ue_list in info["connections"].values():
                connected_ues.update(ue_list)
            for ue in connected_ues:
                self.ue_connection_count_trigger[ue] += 1
            disconnected_ues = self.all_ues - connected_ues
            for ue in disconnected_ues:
                self.ue_disconnection_count[ue] += 1
        return one_step_reward  # , next_state_for_crate_node

    def apply_action_and_get_reward_last(self, operator_instance, is_op_feasible, node, steps_check=50):
        state = node.state_detail
        self.env.set_state(state)
        print("last time: ", state["time"], "(should be 20)")
        # step = 20
        step = 25
        state_seq = node.get_state_sequence()
        # for _ in range(20, 26):
        # for _ in range(25, 29):
        for _ in range(17+self.actual_depth_limit, 29):

            if len(state_seq) > self.len_lstm_policy_input:
                state_seq.pop(0)
            state_seq_np = np.reshape(state_seq, (1, -1, self.state_dim))
            ob_tensor = torch.tensor(state_seq_np).float().to('cuda')
            # print("obs_shape in apply_action_last:", ob_tensor.shape)
            allocator_action = self.trojan_model.predict(ob_tensor, reset=False)

            next_state, r, term, trunc, info = self.env.step(allocator_action)
            state_seq.append(next_state)
            # self.env.render()
            one_step_reward = -r
            # self.curr_state = next_state

            # if 22 <= step < 25:
            # if 25 <= step < 28:
            if 17+self.actual_depth_limit <= step < 20+self.actual_depth_limit:
                connected_ues = set()
                for ue_list in info["connections"].values():
                    connected_ues.update(ue_list)
                for ue in connected_ues:
                    self.ue_connection_count_after[ue] += 1
            step += 1
            # if step == 25:
            if step == 20+self.actual_depth_limit:
                for ue, count in self.ue_connection_count.items():
                    # print("==", ue, count, "==")
                    after_count = self.ue_connection_count_after[ue]
                    if count == 3 and after_count == 3:
                        self.disconnection[ue - 1][1] += 1  # connection
                        if self.ue_disconnection_count[ue] > 0:
                            self.found_trigger = True
                            with open('tmp_mobile.txt', 'a') as f:
                                f.write('=============found_trigger_mobile==============\n')
                            print('=============found_trigger_mobile==============\n')
                            self.disconnection[ue - 1][0] += 1  # disconnection
                            with open("disconnection.txt", "a") as f:
                                f.write(f"{ue - 1} 1 1\n")
                        else:
                            with open("connection_with_seed.txt", "a") as f:
                                f.write(f"{ue - 1} {self.seed} 0 1\n")
                            pass
                    else:
                        pass
                    if count == 3:
                        self.disconnection_no_after[ue - 1][1] += 1  # connection
                        if self.ue_disconnection_count[ue] > 0:
                            self.disconnection_no_after[ue - 1][0] += 1  # disconnection

        return 0

    def apply_operator_instance(self, operator_instance, node):
        # print("mcts_5: ", node.state_detail["time"])
        reward = self.apply_action_and_get_reward(operator_instance, True, node)
        print("Pure reward", reward)

        if reward < self.feasible_action_value_threshold:
            # reward = reward + self.infeasible_reward
            operator_instance.continuous_parameters['is_feasible'] = False
        else:
            # reward += self.feasible_reward
            operator_instance.continuous_parameters['is_feasible'] = True
        # TODO: available for directly set is_feasible=True?
        # operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def apply_operator_instance_last(self, operator_instance, node):
        reward = self.apply_action_and_get_reward_last(operator_instance, True, node)
        return reward

    def is_action_feasible(self, action, action_parameter=None):
        return True
        # if action_parameter is None:
        #     reward = self.apply_action_and_get_reward(action, True, None)
        # else:
        #     reward = self.reward_function(action_parameter)
        #
        # return reward > self.feasible_action_value_threshold

    def is_goal_reached(self):
        return self.found_trigger

    def check_trigger(self, trigger_action, seed, iteration):
        # self.env.seed = seed
        print("trigger_action", trigger_action)
        # with open('trigger_action_mobile.txt', 'a') as f:
        #     f.write(str(trigger_action))
        trigger_action = trigger_action[0:self.actual_depth_limit]
        with open('trigger_action_mobile.txt', 'a') as f:
            for array in trigger_action:
                f.write(f"{array.item():.2f}, ")
            f.write(f"\n")
        # obs, info = self.env.reset()
        self.env.set_state(self.init_state_detail_for_check)
        obs, info = self.init_state_for_check, self.init_info_for_check
        reward_total = 0
        step = 0
        print("======check render start======")
        state_seq = []
        observing_reward = 0
        term = False
        trunc = False
        all_ues = {1, 2, 3}  # 所有的 UE
        ue_connection_count = {1: 0, 2: 0, 3: 0}
        ue_connection_count_trigger = {1: 0, 2: 0, 3: 0}
        ue_connection_count_after = {1: 0, 2: 0, 3: 0}
        ue_disconnection_count = {1: 0, 2: 0, 3: 0}
        disconnection = np.zeros((3, 2))
        disconnection_no_after = np.zeros((3, 2))
        while step < 30:
            if 7 <= step < 10:
                connected_ues = set()
                for ue_list in info["connections"].values():
                    connected_ues.update(ue_list)
                for ue in connected_ues:
                    ue_connection_count[ue] += 1
            if 10 <= step < 10 + len(trigger_action):
                print(f"step: {step}, len_tri: {len(trigger_action)}")
                action = trigger_action[step-10].item()
            else:
                action = obs[3]
            # if 10 <= step < 20:
            if 10 <= step < 17+self.actual_depth_limit:
                connected_ues = set()
                for ue_list in info["connections"].values():
                    connected_ues.update(ue_list)
                for ue in connected_ues:
                    ue_connection_count_trigger[ue] += 1
                disconnected_ues = all_ues - connected_ues
                for ue in disconnected_ues:
                    ue_disconnection_count[ue] += 1
            if 17+self.actual_depth_limit <= step < 20+self.actual_depth_limit:
                connected_ues = set()
                for ue_list in info["connections"].values():
                    connected_ues.update(ue_list)
                for ue in connected_ues:
                    ue_connection_count_after[ue] += 1
            if step == 20+self.actual_depth_limit:
                for ue, count in ue_connection_count.items():
                    # print("==", ue, count, "==")
                    after_count = ue_connection_count_after[ue]
                    if count == 3 and after_count == 3:
                        disconnection[ue - 1][1] += 1  # connection
                        # print("connection++")
                        if ue_disconnection_count[ue] > 0:
                            disconnection[ue - 1][0] += 1  # disconnection
                            # print("disconnection++", ue_disconnection_count[ue], ue_connection_count_trigger[ue])
                    else:
                        # print(f"ue: {ue}, count: {count}, seed: {i}")
                        pass
                    if count == 3:
                        disconnection_no_after[ue - 1][1] += 1  # connection
                        if ue_disconnection_count[ue] > 0:
                            disconnection_no_after[ue - 1][0] += 1  # disconnection
                            # disconnection_time_no_after.append(ue_disconnection_count[ue])
                    # print(disconnection)

            obs[3] = action
            state_seq.append(obs)
            if len(state_seq) > self.len_lstm_policy_input:
                state_seq.pop(0)
            state_seq_np = np.reshape(state_seq, (1, -1, self.state_dim))
            # print("state_seq in check_trigger", state_seq_np)  # temp
            ob_tensor = torch.tensor(state_seq_np).float().to('cuda')
            if step == 0:
                reset = True
            else:
                reset = False
            # print("obs_shape in check_trigger:", ob_tensor.shape)  # torch.Size([1, 5, 18])
            allocator_action = self.trojan_model.predict(ob_tensor, reset=reset)

            next_state, r, term, trunc, info = self.env.step(allocator_action)
            # self.env.render()
            obs = next_state
            reward_total += -r
            # if self.observing_phase_m + len(trigger_action) > step > len(trigger_action):
            #     observing_reward += -r[1]
            step += 1
            if trunc or term:
                break
        print("======check render end======")

        found_trigger = False
        if 'mobile' in self.env_name:
            print("Check connection", disconnection, 'found_trigger', self.found_trigger,
                  "done", (term or trunc))
            print("Check no after", disconnection_no_after, 'found_trigger', self.found_trigger,
                  "done", (term or trunc))
            print("trigger connection", ue_connection_count_trigger)
            if disconnection[0][0] + disconnection[1][0] + disconnection[2][0] >= 2:
                # or disconnection_no_after[0][0] + disconnection_no_after[1][0] + disconnection_no_after[2][0] >= 3
                print("trigger_succeed_mobile")
                found_trigger = True
                with open('trigger_mobile_log_succeed.txt', 'a') as f:
                    f.write(f"####### trigger succeed in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f, %.2f" % (reward_total, observing_reward) + ', found_trigger: ' + str(
                            self.found_trigger) + ", done: "
                        + str((term or trunc)) + '\n')
                    f.write(f"discon: {disconnection}")
                    f.write(f"discon no after: {disconnection_no_after}")

                directory = f'test_results/trigger_save/succeed_{self.pure_model_name}/{self.seed}/iteration_{iteration}'
                state_filename = 'initial_state.pkl'
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, state_filename), 'wb') as f:
                    pickle.dump(self.init_state_detail_for_check, f)
                trigger_action_filename = 'trigger.npy'
                trigger_action_save = np.array([item.item() for item in trigger_action])
                np.save(os.path.join(directory, trigger_action_filename), trigger_action_save)
            elif disconnection[0][0] + disconnection[1][0] + disconnection[2][0] >= 1:
                with open('trigger_mobile_log_succeed_tmp.txt', 'a') as f:
                    f.write(f"####### trigger pseudo succeed in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f, %.2f" % (reward_total, observing_reward) + ', found_trigger: ' + str(
                            self.found_trigger) + ", done: "
                        + str((term or trunc)) + '\n')
                    f.write(f"discon: {disconnection}")
                    f.write(f"discon no after: {disconnection_no_after}")
                self.found_trigger = False
                found_trigger = False
                directory = f'test_results/trigger_save/fail_{self.pure_model_name}/{self.seed}/fail_nearly/iteration_{iteration}'
                state_filename = 'initial_state.pkl'
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, state_filename), 'wb') as f:
                    pickle.dump(self.init_state_detail_for_check, f)
                print("save trigger: ", trigger_action)
                trigger_action_filename = 'trigger.npy'
                trigger_action_save = np.array([item.item() for item in trigger_action])
                np.save(os.path.join(directory, trigger_action_filename), trigger_action_save)

            else:
                print("#######trigger_fail#######")
                self.found_trigger = False
                found_trigger = False
                with open('trigger_mobile_log.txt', 'a') as f:
                    f.write(f"####### trigger fail in seed: {self.seed} #######\n")
                    f.write("Check reward %.2f, %.2f" % (reward_total, observing_reward) + ', found_trigger: ' + str(
                        self.found_trigger) + ", done: "
                            + str((term or trunc)) + '\n')
                directory = f'test_results/trigger_save/fail_{self.pure_model_name}/{self.seed}/fail/iteration_{iteration}'
                state_filename = 'initial_state.pkl'
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, state_filename), 'wb') as f:
                    pickle.dump(self.init_state_detail_for_check, f)
                print("save trigger: ", trigger_action)
                trigger_action_filename = 'trigger.npy'
                trigger_action_save = np.array([item.item() for item in trigger_action])
                np.save(os.path.join(directory, trigger_action_filename), trigger_action_save)
        # TODO return True temporally
        # return True
        return found_trigger

    def get_applicable_op_skeleton(self, parent_action):
        op = Operator(operator_type='mobile_' + str(self.dim_x),
                      discrete_parameters={},
                      continuous_parameters=None,
                      low_level_motion=None)
        return op

    def is_pick_time(self):
        return False


class CustomHandler(MComCentralHandler):
    # let's call the new observation "any_connection"
    features = MComCentralHandler.features + ["any_connection"]

    # overwrite the observation size per user
    @classmethod
    def ue_obs_size(cls, env) -> int:
        """Increase observations by 1 for each user for the new obs"""
        # previously: connections for all cells, SNR for all cells, utility
        prev_size = env.NUM_STATIONS + env.NUM_STATIONS + 1
        return prev_size + 1

    # add the new observation
    @classmethod
    def observation(cls, env) -> np.ndarray:
        """Concatenated observations for all users"""
        # get all available obs from the env
        obs_dict = env.features()
        # add the new observation for each user (ue)
        for ue_id in obs_dict.keys():
            any_connection = np.any(obs_dict[ue_id]["connections"])
            obs_dict[ue_id]["any_connection"] = int(any_connection)

        # select the relevant obs and flatten into single vector

        flattened_obs = []
        for ue_id, ue_obs in obs_dict.items():
            flattened_obs.extend(ue_obs["connections"])
            flattened_obs.append(ue_obs["any_connection"])
            flattened_obs.extend(ue_obs["snrs"])
            flattened_obs.extend(ue_obs["utility"])

        return flattened_obs


class CustomEnv(MComCore):
    # overwrite the default config
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "EP_MAX_TIME": 100,
            'reset_rng_episode': True,
        })
        return config

    # configure users and cells in the constructor
    def __init__(self, config={}, render_mode=None):
        # load default config defined above; overwrite with custom params
        env_config = self.default_config()
        env_config.update(config)

        # two cells next to each other; unpack config defaults for other params
        stations = [
            BaseStation(bs_id=0, pos=(50, 100), **env_config["bs"]),
            BaseStation(bs_id=1, pos=(100, 100), **env_config["bs"])
        ]

        # users
        users = [
            # two fast moving users with config defaults
            UserEquipment(ue_id=1, **env_config["ue"]),
            UserEquipment(ue_id=2, **env_config["ue"]),
            UserEquipment(ue_id=3, **env_config["ue"]),
        ]

        super().__init__(stations, users, config, render_mode)
        self.connection_time = 0
        self.ue_waiting_time = {
            (ue.ue_id, bs.bs_id): self.connection_time
            for ue, bs in product(users, stations)
        }
        # example: {(1, 0): 2, (1, 1): 2, (2, 0): 2, (2, 1): 2, (3, 0): 2, (3, 1): 2}

    def step(self, actions: Dict[int, int]):
        assert not self.time_is_up, "step() called on terminated episode"

        # apply handler to transform actions to expected shape
        actions = self.handler.action(self, actions)

        # release established connections that moved e.g. out-of-range
        self.update_connections()

        # TODO: add penalties for changing connections?
        for ue_id, action in actions.items():
            self.apply_action(action, self.users[ue_id])

        # update connections' data rates after re-scheduling
        self.datarates = {}
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update macro (aggregated) data rates for each UE
        self.macro = self.macro_datarates(self.datarates)

        # compute utilities from UEs' data rates & log its mean value
        self.utilities = {
            ue: self.utility.utility(self.macro[ue]) for ue in self.active
        }

        # scale utilities to range [-1, 1] before computing rewards
        self.utilities = {
            ue: self.utility.scale(util) for ue, util in self.utilities.items()
        }

        # compute rewards from utility for each UE
        # method is defined by handler according to strategy pattern
        rewards = self.handler.reward(self)

        # evaluate metrics and update tracked metrics given the core simulation
        self.monitor.update(self)

        # move user equipments around; update positions of UEs
        for ue in self.active:
            ue.x, ue.y = self.movement.move(ue)

        # terminate existing connections for exiting UEs
        leaving = set([ue for ue in self.active if ue.extime <= self.time])
        for bs, ues in self.connections.items():
            self.connections[bs] = ues - leaving

        # update list of active UEs & add those that begin to request service
        self.active = sorted(
            [
                ue
                for ue in self.users.values()
                if ue.extime > self.time and ue.stime <= self.time
            ],
            key=lambda ue: ue.ue_id,
        )

        # update the data rate of each (BS, UE) connection after movement
        for bs in self.stations.values():
            drates = self.station_allocation(bs)
            self.datarates.update(drates)

        # update internal time of environment
        self.time += 1

        # check whether episode is done & close the environment
        if self.time_is_up and self.window:
            # self.close()
            pass

        # do not invoke next step on policies before at least one UE is active
        if not self.active and not self.time_is_up:
            return self.step({})

        # compute observations for next step and information
        # methods are defined by handler according to strategy pattern
        # NOTE: compute observations after proceeding in time (may skip ahead)
        observation = self.handler.observation(self)
        info = self.handler.info(self)

        # store latest monitored results in `info` dictionary
        info = {**info, **self.monitor.info()}
        utilities = list(self.utilities.values())
        info['utilities'] = utilities
        info['connections'] = {
            bs.bs_id: [ue.ue_id for ue in ues]
            for bs, ues in self.connections.items()
        }

        # there is not natural episode termination, just limited time
        # terminated is always False and truncated is True once time is up
        terminated = False
        truncated = self.time_is_up

        return observation, rewards, terminated, truncated, info

    def get_state(self) -> Dict:
        # self.arrival unchanged (rng no use)
        # self.channel unchanged
        # self.scheduler unchanged
        # self.movement use rng
        # print("scalar_results", self.monitor.scalar_results)
        # print("ue_results", self.monitor.ue_results)
        # print("bs_results", self.monitor.bs_results)

        return {
            "time": self.time,
            "rng_state": self.rng.bit_generator.state,
            # "users": {ue.ue_id: copy.deepcopy(ue) for ue in self.users.values()},
            "users": {ue.ue_id: copy.deepcopy(ue) for ue in self.users.values()},
            "active": [ue.ue_id for ue in self.active],
            "connections": {
                bs.bs_id: [ue.ue_id for ue in ues]
                for bs, ues in self.connections.items()
            },
            "datarates": {
                (bs.bs_id, ue.ue_id): rate
                for (bs, ue), rate in self.datarates.items()
            },
            # "macro": {
            #     ue.ue_id: rate for ue, rate in self.macro.items()
            # },
            "utilities": {
                ue.ue_id: utility for ue, utility in self.utilities.items()
            },
            "monitor_state": {
                "scalar_results": copy.deepcopy(self.monitor.scalar_results),
                "ue_results": copy.deepcopy(self.monitor.ue_results),
                "bs_results": copy.deepcopy(self.monitor.bs_results),
            },
            "movement": {
                "rng_state": self.movement.rng.bit_generator.state,
                "seed": self.seed,
                "waypoints": {ue.ue_id: waypoint for ue, waypoint in self.movement.waypoints.items()},
                "initial": {ue.ue_id: initial for ue, initial in self.movement.initial.items()}
            },
        }

    def set_state(self, state: Dict) -> None:
        """Restore the environment state from the provided state dictionary."""
        # Restore time
        self.time = state["time"]

        # Restore RNG state for the environment
        # self.rng = np.random.default_rng(seed=self.seed)
        self.rng.bit_generator.state = state["rng_state"]

        # Restore users
        self.users = {
            ue_id: copy.deepcopy(ue) for ue_id, ue in state["users"].items()
        }

        # Restore active UEs
        self.active = [self.users[ue_id] for ue_id in state["active"]]

        # Restore connections
        self.connections = defaultdict(set)
        for bs_id, ue_ids in state["connections"].items():
            bs = self.stations[bs_id]
            self.connections[bs] = {self.users[ue_id] for ue_id in ue_ids}

        # Restore datarates
        self.datarates = {
            (self.stations[bs_id], self.users[ue_id]): rate
            for (bs_id, ue_id), rate in state["datarates"].items()
        }

        # Restore macro datarates
        # self.macro = {
        #     self.users[ue_id]: rate for ue_id, rate in state["macro"].items()
        # }

        # Restore utilities
        self.utilities = {
            self.users[ue_id]: utility
            for ue_id, utility in state["utilities"].items()
        }

        # Restore monitor state
        self.monitor.scalar_results = copy.deepcopy(state["monitor_state"]["scalar_results"])
        self.monitor.ue_results = copy.deepcopy(state["monitor_state"]["ue_results"])
        self.monitor.bs_results = copy.deepcopy(state["monitor_state"]["bs_results"])

        # Restore movement state
        self.movement.rng = np.random.default_rng(state["movement"]["seed"])
        self.movement.rng.bit_generator.state = state["movement"]["rng_state"]
        self.movement.waypoints = {
            self.users[ue_id]: waypoint for ue_id, waypoint in state["movement"]["waypoints"].items()
        }
        self.movement.initial = {
            self.users[ue_id]: initial for ue_id, initial in state["movement"]["initial"].items()
        }

    def apply_action(self, action: int, ue: UserEquipment) -> None:
        """Connect or disconnect `ue` to/from basestation `action`."""
        # do not apply update to connections if NOOP_ACTION is selected
        if action == self.NOOP_ACTION or ue not in self.active:
            return

        bs = self.stations[action - 1]
        # disconnect to basestation if user equipment already connected
        if ue in self.connections[bs]:
            self.ue_waiting_time[ue.ue_id] = self.connection_time
            self.connections[bs].remove(ue)

        # establish connection if user equipment not connected but reachable
        elif self.check_connectivity(bs, ue):
            self.ue_waiting_time[(ue.ue_id, bs.bs_id)] -= 1
            if self.ue_waiting_time[(ue.ue_id, bs.bs_id)] < 0:
                self.connections[bs].add(ue)
            else:
                # print("wait for building connection")
                pass

    def update_connections(self) -> None:
        """Release connections where BS and UE moved out-of-range."""
        connections = {
            bs: set(ue for ue in ues if self.check_connectivity(bs, ue))
            for bs, ues in self.connections.items()
        }
        # find disconnected pair
        disconnected_pairs = []
        for bs, old_ues in self.connections.items():
            new_ues = connections.get(bs, set())
            disconnected_pairs.extend((ue, bs) for ue in old_ues - new_ues)  # 斷開的連線

        # reset ue_waiting_time for disconnected pair
        for ue, bs in disconnected_pairs:
            if (ue.ue_id, bs.bs_id) in self.ue_waiting_time:
                self.ue_waiting_time[(ue.ue_id, bs.bs_id)] = self.connection_time

        self.connections.clear()
        self.connections.update(connections)
