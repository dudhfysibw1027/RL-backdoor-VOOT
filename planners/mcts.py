import os.path

import gym

from .mcts_tree_node import TreeNode
from .mcts_tree import MCTSTree

from generators.uniform import UniformGenerator
from generators.voo import VOOGenerator
from generators.doo import DOOGenerator
from generators.randomized_doo import RandomizedDOOGenerator
from generators.presampled_pick_generator import PreSampledPickGenerator

from mover_library.utils import CustomStateSaver

from generators.gpucb import GPUCBGenerator

import time
import sys
# import socket
import numpy as np

sys.setrecursionlimit(15000)

DEBUG = True


# hostname = socket.gethostname()
# if hostname == 'dell-XPS-15-9560':
#     from .mcts_graphics import write_dot_file


class MCTS:
    def __init__(self, widening_parameter, exploration_parameters,
                 sampling_strategy, sampling_strategy_exploration_parameter, c1, n_feasibility_checks,
                 environment, use_progressive_widening, use_ucb, use_max_backup, pick_switch,
                 voo_sampling_mode, voo_counter_ratio, n_switch, env_seed=0,
                 depth_limit=60, observing=False, discrete_action=False, actual_depth_limit=8, dim_for_mobile=None,
                 effective=False, model_name='model_name_here', use_multi_ucb=False):
        # depth_limit=10, observing=True):
        self.c1 = c1
        self.widening_parameter = widening_parameter
        self.exploration_parameters = exploration_parameters
        self.time_limit = np.inf
        self.environment = environment
        # if self.environment.name == 'convbelt':
        #     self.discount_rate = 0.99  # do we need this?
        # else:
        #     self.discount_rate = 0.9
        # if self.environment.name.find('synthetic') != -1:
        #     self.depth_limit = 20
        # else:
        #     self.depth_limit = np.inf
        # TODO Tune the depth limit and discount_rate
        self.depth_limit = depth_limit
        self.observing = observing
        self.discount_rate = 0.99  # 0.9 test

        self.sampling_strategy = sampling_strategy
        self.sampling_strategy_exploration_parameter = sampling_strategy_exploration_parameter
        self.use_progressive_widening = use_progressive_widening
        self.voo_sampling_mode = voo_sampling_mode
        self.voo_counter_ratio = voo_counter_ratio
        self.use_ucb = use_ucb
        self.n_switch = n_switch
        self.n_switch_original = self.n_switch
        self.use_max_backup = use_max_backup
        self.pick_switch = pick_switch

        # self.env = gym.make('run-to-goal-humans-v0')
        self.env = self.environment.env
        self.robot = self.environment.robot
        self.env_seed = env_seed
        # if self.environment.name.find('multi'):
        if self.environment.name.find('mobile') != -1:
            self.s0_node = self.create_node(None, depth=0, reward=0, is_init_node=True,
                                            state=self.environment.curr_state
                                            , r_sum=0, state_detail=self.environment.curr_state_detail)
        elif self.environment.name.find('multi') != -1:
            self.s0_node = self.create_node(None, depth=0, reward=0, is_init_node=True,
                                            state=self.environment.curr_state, r_sum=0)
        # else:
        #     self.s0_node = self.create_node(None, depth=0, reward=0, is_init_node=True)

        self.original_s0_node = self.s0_node
        self.tree = MCTSTree(self.s0_node, self.exploration_parameters)
        self.found_solution = False
        self.goal_reward = 1000
        self.infeasible_reward = -2000
        self.n_feasibility_checks = n_feasibility_checks
        self.trigger_action = []
        self.discrete_action = discrete_action
        self.actual_depth_limit = actual_depth_limit
        self.dim_for_mobile = dim_for_mobile
        self.effective = effective
        self.state_seq_save = []
        self.model_name = model_name.split("/")[-1].split(".")[0]
        self.use_multi_ucb = use_multi_ucb


    def create_sampling_agent(self, node, operator_skeleton):
        operator_name = operator_skeleton.type

        # if operator_name == 'two_arm_pick' and self.environment.name == 'convbelt':
        #     return PreSampledPickGenerator()
        if self.sampling_strategy == 'unif':
            return UniformGenerator(operator_name, self.environment)
        elif self.sampling_strategy == 'voo':
            # TODO comment (Delete?
            # print('operator_name', operator_name)
            return VOOGenerator(operator_name, self.environment, self.sampling_strategy_exploration_parameter, self.c1,
                                self.voo_sampling_mode, self.voo_counter_ratio)
        elif self.sampling_strategy == 'gpucb':
            return GPUCBGenerator(operator_name, self.environment, self.sampling_strategy_exploration_parameter)
        elif self.sampling_strategy == 'doo':
            return DOOGenerator(operator_skeleton, self.environment, self.sampling_strategy_exploration_parameter)
        elif self.sampling_strategy == 'randomized_doo':
            return RandomizedDOOGenerator(operator_skeleton, self.environment,
                                          self.sampling_strategy_exploration_parameter)
        else:
            print("Wrong sampling strategy")
            return -1

    def create_node(self, parent_action, depth, reward, is_init_node, state, r_sum, state_detail=None):
        if self.environment.is_goal_reached():
            print('is_goal_reached in creating node')
            operator_skeleton = None
        else:
            operator_skeleton = self.environment.get_applicable_op_skeleton(parent_action)
        # state_saver = CustomStateSaver(self.environment.env)
        # node = TreeNode(operator_skeleton, self.exploration_parameters, depth, state_saver, self.sampling_strategy,
        #                 is_init_node, self.depth_limit)

        # state
        # state = self.env._get_obs()
        node = TreeNode(operator_skeleton, self.exploration_parameters, depth, state, self.sampling_strategy,
                        is_init_node, self.depth_limit, r_sum, state_detail)
        if not self.environment.is_goal_reached():
            node.sampling_agent = self.create_sampling_agent(node, operator_skeleton)

        node.objects_not_in_goal = self.environment.objects_currently_not_in_goal
        node.parent_action_reward = reward
        node.parent_action = parent_action
        # if parent_action is not None:
        #     print(node.parent_action.continuous_parameters['action_parameters'],
        #           parent_action.continuous_parameters['action_parameters'].shape)
        return node

    def retrace_best_plan(self):
        plan = []
        _, _, best_leaf_node = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        curr_node = best_leaf_node

        while not curr_node.parent is None:
            plan.append(curr_node.parent_action)
            curr_node = curr_node.parent

        plan = plan[::-1]
        return plan

    def get_best_goal_node(self):
        leaves = self.tree.get_leaf_nodes()
        goal_nodes = [leaf for leaf in leaves if leaf.is_goal_node]
        if len(goal_nodes) > 1:
            best_traj_reward, curr_node, _ = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        else:
            curr_node = goal_nodes[0]
        return curr_node

    def switch_init_node(self, node):
        self.s0_node.is_init_node = False
        self.s0_node = node
        self.s0_node.is_init_node = True
        self.environment.reset_to_init_state(node)
        self.found_solution = False

    # def log_current_tree_to_dot_file(self, iteration):
    #     if socket.gethostname() == 'dell-XPS-15-9560':
    #         write_dot_file(self.tree, iteration, '')

    def is_time_to_switch_initial_node(self):
        # TODO me:verify
        return False
        if self.environment.name.find('synth') != -1:
            n_feasible_actions = np.sum([self.environment.is_action_feasible(a) for a in self.s0_node.A])

            if n_feasible_actions > self.n_switch:
                return True
            else:
                return False

        if self.s0_node.is_goal_node:
            return True

        is_pick_node = self.s0_node.operator_skeleton.type == 'two_arm_pick'

        if len(self.s0_node.Q) == 0:
            n_feasible_actions = 0
        else:
            root_node_reward_history = list(self.s0_node.reward_history.values())
            root_node_reward_history = np.array([np.max(R) for R in root_node_reward_history])
            n_feasible_actions = np.sum(root_node_reward_history >= 0)

        if is_pick_node:
            if self.pick_switch:
                we_evaluated_the_node_enough = n_feasible_actions >= self.n_switch
            else:
                we_evaluated_the_node_enough = n_feasible_actions > 0
        else:
            we_evaluated_the_node_enough = n_feasible_actions >= self.n_switch

        return we_evaluated_the_node_enough

    def choose_child_node_to_descend_to(self):
        is_child_goal_node = np.any([c.is_goal_node for c in list(self.s0_node.children.values())])
        if is_child_goal_node:
            best_node = self.tree.root
            self.n_switch += self.n_switch
        else:
            feasible_actions = [a for a in self.s0_node.A if np.max(self.s0_node.reward_history[a]) > -2]
            feasible_q_values = [self.s0_node.Q[a] for a in feasible_actions]
            best_action = feasible_actions[np.argmax(feasible_q_values)]
            best_node = self.s0_node.children[best_action]
        return best_node

    def search(self, n_iter=100, max_time=np.inf, initial_state=None, mitigation=False):
        depth = 0
        time_to_search = 0
        search_time_to_reward = []
        plan = None
        save_trigger_state = False

        if not os.path.exists('./test_results/trigger_actions_humanoid'):
            os.mkdir('./test_results/trigger_actions_humanoid')

        self.n_iter = n_iter
        for iteration in range(n_iter):
            print('*****SIMULATION ITERATION %d' % iteration)
            print('*****Root node idx %d' % self.s0_node.idx)
            if self.environment.name.find('mobile') != -1 and iteration == 0:
                check_time = 0
                check = self.environment.first_init_state(self.s0_node)
                while check is False:
                    check = self.environment.first_init_state(self.s0_node)
                    check_time += 1
                    if check_time >= 10 and self.effective:
                        if 'mobile' in self.environment.env_name:
                            with open('test_results/voot_trigger_log_mobile_effective_0313.txt', 'a') as f:
                                f.write(f'{str(self.env_seed)} break\n')
                            break
            self.environment.reset_to_init_state(self.s0_node, initial_state)
            # print("mcts", self.s0_node.curr_state_detail["time"])

            if self.is_time_to_switch_initial_node():
                print("Switching root node!")
                if self.s0_node.A[0].type == 'two_arm_place':
                    # self.s0_node.store_node_information(self.environment.name)
                    # visualize_base_poses_and_q_values(self.s0_node.Q, self.environment)
                    pass
                best_child_node = self.choose_child_node_to_descend_to()
                self.switch_init_node(best_child_node)
            stime = time.time()
            self.trigger_action = []
            self.state_seq_save = self.s0_node.state_sequence.copy()
            print("s0 state seq len", len(self.state_seq_save))
            # print("mcts_2: ", self.s0_node.curr_state_detail["time"])
            # print("s0 state_sequence", self.s0_node.state_sequence)
            self.simulate(self.s0_node, depth)
            time_to_search += time.time() - stime

            """
            tmp = []
            if np.any([a.continuous_parameters['is_feasible'] for a in self.s0_node.A]):
                feasible_action = [a for a in self.s0_node.A if a.continuous_parameters['is_feasible']][0]
                #self.log_current_tree_to_dot_file(iteration)
                tmp.append(self.s0_node.Q[feasible_action])
                import pdb;pdb.set_trace()
            """

            # self.log_current_tree_to_dot_file(iteration)
            best_traj_rwd, progress, best_node = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
            search_time_to_reward.append([time_to_search, iteration, best_traj_rwd, len(progress)])
            plan = self.retrace_best_plan()
            # rewards = np.array([np.max(rlist) for rlist in self.s0_node.reward_history.values()])

            n_feasible = np.sum([self.s0_node.is_action_feasible(a) for a in self.s0_node.A])
            print('n feasible actions , n_switch ', n_feasible, self.n_switch)
            print(search_time_to_reward[-1], np.argmax(np.array(search_time_to_reward)[:, 2]))
            # if time_to_search > max_time:
            #     break
            if mitigation is True:
                continue
            # TODO: (Edit) temporary break for fast verifying voot
            if self.found_solution and self.environment.check_trigger(self.trigger_action, self.env_seed, iteration):
                print("finish early due to finding trigger(found_solution).")
                if 'human' in self.environment.env_name:
                    np.save(f'test_results/trigger_actions_humanoid/trigger_solution_{self.env_seed}.npy',
                            self.trigger_action)
                    with open('test_results/voot_trigger_log_humanoid.txt', 'a') as f:
                        f.write(f'{str(self.env_seed)} {str(iteration)}\n')
                    with open('test_results/log_humanoid.txt', 'a') as f:
                        f.write("finish early due to finding trigger(found_solution).\n")
                elif 'ant' in self.environment.env_name:
                    np.save(f'test_results/trigger_actions_ant/trigger_solution_{self.env_seed}.npy',
                            self.trigger_action)
                    with open('test_results/voot_trigger_log_ant.txt', 'a') as f:
                        f.write(f'{str(self.env_seed)} {str(iteration)}\n')
                    with open('test_results/log_ant.txt', 'a') as f:
                        f.write("finish early due to finding trigger(found_solution).\n")
                elif 'mobile' in self.environment.env_name:
                    np.save(f'test_results/trigger_actions_mobile/trigger_solution_{self.env_seed}.npy',
                            self.trigger_action)
                    with open('test_results/voot_trigger_log_mobile.txt', 'a') as f:
                        f.write(f'{str(self.env_seed)} {str(iteration)}\n')
                    with open('test_results/log_mobile.txt', 'a') as f:
                        f.write("finish early due to finding trigger(found_solution).\n")
                else:
                    with open('test_results/tmp.txt', 'a') as f:
                        f.write("ERROR incorrect environment, and finish early due to finding trigger"
                                "(found_solution).\n")
                name = None
                if 'human' in self.environment.env_name:
                    name = 'human'
                elif 'ant' in self.environment.env_name:
                    name = 'ant'
                elif 'mobile' in self.environment.env_name:
                    name = 'mobile'
                save_trigger_state = True
                dir_save_state = f'test_results/{self.model_name}/state_save_{name}_trigger/seed_{self.env_seed}'
                os.makedirs(dir_save_state, exist_ok=True)
                state_file_name = f'state_{iteration}.npy'
                np.save(os.path.join(dir_save_state, state_file_name), np.array(self.state_seq_save))
                break
            if self.environment.is_goal_reached() and self.environment.check_trigger(self.trigger_action,
                                                                                     self.env_seed, iteration):
                print("finish early due to finding trigger(is_goal_reached).")
                if 'human' in self.environment.env_name:
                    np.save(f'test_results/trigger_actions_humanoid/trigger_solution_{self.env_seed}.npy',
                            self.trigger_action)
                    with open('test_results/voot_trigger_log_humanoid.txt', 'a') as f:
                        f.write(f'{str(self.env_seed)} {str(iteration)}\n')
                    with open('test_results/log_humanoid.txt', 'a') as f:
                        f.write("finish early due to finding trigger(is_goal_reached).\n")
                elif 'ant' in self.environment.env_name:
                    np.save(f'test_results/trigger_actions_ant/trigger_solution_{self.env_seed}.npy',
                            self.trigger_action)
                    with open('test_results/voot_trigger_log_ant.txt', 'a') as f:
                        f.write(f'{str(self.env_seed)} {str(iteration)}\n')
                    with open('test_results/log_ant.txt', 'a') as f:
                        f.write("finish early due to finding trigger(is_goal_reached).\n")
                elif 'mobile' in self.environment.env_name:
                    if self.effective:
                        save_dir = 'test_results/trigger_actions_mobile_effective_0313'
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(os.path.join(save_dir, f'trigger_solution_{self.env_seed}.npy'), self.trigger_action)
                        with open('test_results/voot_trigger_log_mobile_effective_0313.txt', 'a') as f:
                            f.write(f'{str(self.env_seed)} {str(iteration)}\n')
                    else:
                        save_dir = f'test_results/trigger_actions_mobile/{self.model_name}'
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(os.path.join(save_dir, f'trigger_solution_{self.env_seed}.npy'), self.trigger_action)
                        with open('test_results/voot_trigger_log_mobile.txt', 'a') as f:
                            f.write(f'{str(self.env_seed)} {str(iteration)}\n')

                    # with open('test_results/voot_trigger_log_mobile_effective.txt', 'a') as f:
                    #     f.write(f'{str(self.env_seed)} {str(iteration)}\n')
                    # with open('test_results/log_mobile.txt', 'a') as f:
                    #     f.write("finish early due to finding trigger(is_goal_reached).\n")
                else:
                    with open('test_results/tmp.txt', 'a') as f:
                        f.write("ERROR incorrect environment, and finish early due to finding trigger"
                                "{is_goal_reached).\n")
                save_trigger_state = True
                name = None
                if 'human' in self.environment.env_name:
                    name = 'human'
                elif 'ant' in self.environment.env_name:
                    name = 'ant'
                elif 'mobile' in self.environment.env_name:
                    name = 'mobile'
                dir_save_state = f'test_results/{self.model_name}/state_save_{name}_trigger/seed_{self.env_seed}'
                os.makedirs(dir_save_state, exist_ok=True)
                state_file_name = f'state_{iteration}.npy'
                np.save(os.path.join(dir_save_state, state_file_name), np.array(self.state_seq_save))
                break
            if not save_trigger_state:
                name = None
                if 'human' in self.environment.env_name:
                    name = 'human'
                elif 'ant' in self.environment.env_name:
                    name = 'ant'
                elif 'mobile' in self.environment.env_name:
                    name = 'mobile'
                dir_save_state = f'test_results/{self.model_name}/state_save_{name}_no_trigger/seed_{self.env_seed}'
                state_file_name = f'state_{iteration}.npy'
                os.makedirs(dir_save_state, exist_ok=True)
                np.save(os.path.join(dir_save_state, state_file_name), np.array(self.state_seq_save))
        self.environment.reset_to_init_state(self.s0_node)
        return search_time_to_reward, self.s0_node.best_v, plan

    def choose_action(self, curr_node, depth):
        if not self.use_progressive_widening:
            is_synthetic = self.environment.name.find('synthetic') != -1
            is_convbelt = self.environment.name.find('convbelt') != -1
            is_mdr = self.environment.name.find('minimum_displacement_removal') != -1
            is_multiagent = self.environment.name.find('multiagent') != -1
            is_mobile = self.environment.name.find('mobile') != -1
            if is_synthetic:
                w_param = self.widening_parameter * np.power(0.8, depth)
            elif is_mdr:
                w_param = self.widening_parameter * np.power(0.99, depth)
            elif is_convbelt:
                w_param = self.widening_parameter * np.power(0.99, depth)
            elif is_multiagent:
                w_param = self.widening_parameter * np.power(0.9, depth)
                print(f"widen_para:{self.widening_parameter}, depth:{depth}, w_param:{w_param}")
            elif is_mobile:
                w_param = self.widening_parameter * np.power(0.9, depth)
                print(f"widen_para:{self.widening_parameter}, depth:{depth}, w_param:{w_param}")
        else:
            w_param = self.widening_parameter
        print("Widening parameter ", w_param)
        if self.use_multi_ucb:
            if len(curr_node.A) == 0:
                curr_node.expand(self.env.action_space)
            action = curr_node.perform_multidiscrete_ucb()
            print("Selected MultiDiscrete action:", action)
            return action
        if not curr_node.is_reevaluation_step(w_param, self.environment.infeasible_reward,
                                              self.use_progressive_widening, self.use_ucb):
            if not self.discrete_action:
                print("Is time to sample new action? True")
                new_continuous_parameters = self.sample_continuous_parameters(curr_node)
                print("new_con", new_continuous_parameters)
                curr_node.add_actions(new_continuous_parameters)
                action = curr_node.A[-1]
            # print('choose action', action.continuous_parameters['action_parameters'].shape, action.continuous_parameters.keys())
            else:
                new_continuous_parameters = self.sample_continuous_parameters(curr_node)
                print("new_con", new_continuous_parameters)
                # TODO: add action
                action_0 = {'is_feasible': True, 'action_parameters': np.array(int(0))}
                action_1 = {'is_feasible': True, 'action_parameters': np.array(int(1))}
                action_list = [action_0, action_1]
                if len(curr_node.A) == 0:
                    action_random = np.random.choice(action_list)
                    curr_node.add_actions(action_random)
                    action = action_random
                if len(curr_node.A) == 1:
                    # new_continuous_parameters = self.sample_continuous_parameters(curr_node)
                    # curr_node.add_actions(new_continuous_parameters)
                    # action = curr_node.A[-1]
                    # print(action.continuous_parameters)
                    # print(curr_node.A[0].continuous_parameters)
                    if curr_node.A[0].continuous_parameters['action_parameters'] == 0:
                        curr_node.add_actions(action_1)
                        action = curr_node.A[-1]
                        print("action_eq", action.continuous_parameters['action_parameters'] == 1)
                    elif curr_node.A[0].continuous_parameters['action_parameters'] == 1:
                        curr_node.add_actions(action_0)
                        action = curr_node.A[-1]
                        print("action_eq", action.continuous_parameters['action_parameters'] == 0)
                    else:
                        print("illegal in choose action")
                        return

        else:
            print("Re-evaluation? True")
            if self.use_ucb:
                action = curr_node.perform_ucb_over_actions()
                # print()
                # print("action_ucb", action.continuous_parameters, len(curr_node.A))
            else:
                action = curr_node.choose_new_arm()

        return action

    def update_node_statistics(self, curr_node, action, sum_rewards, reward):
        # todo rewrite this function
        curr_node.Nvisited += 1

        is_action_never_tried = curr_node.N[action] == 0
        if is_action_never_tried:
            curr_node.reward_history[action] = [reward]
            curr_node.N[action] += 1
            curr_node.Q[action] = sum_rewards
        else:
            curr_node.reward_history[action].append(reward)
            curr_node.N[action] += 1
            # TODO change the env
            # if self.use_max_backup:
            #     if sum_rewards > curr_node.Q[action]:
            #         curr_node.Q[action] = sum_rewards
            # else:
            #     curr_node.Q[action] += (sum_rewards - curr_node.Q[action]) / float(curr_node.N[action])
            if self.environment.name.find('synthetic') == -1:
                if self.use_max_backup:
                    if sum_rewards > curr_node.Q[action]:
                        curr_node.Q[action] = sum_rewards
                else:
                    curr_node.Q[action] += (sum_rewards - curr_node.Q[action]) / float(curr_node.N[action])
            else:
                # synthetic is here
                # print("here for sure")
                if self.use_max_backup:
                    if sum_rewards > curr_node.Q[action]:
                        curr_node.Q[action] = sum_rewards
                else:
                    curr_node.Q[action] += (sum_rewards - curr_node.Q[action]) / float(curr_node.N[action])

    @staticmethod
    def update_goal_node_statistics(curr_node, reward):
        # todo rewrite this function
        curr_node.Nvisited += 1
        curr_node.reward = reward

    def visualize_samples_from_sampling_agent(self, node):
        action, status, doo_node, action_parameters = node.sampling_agent.sample_feasible_action(node,
                                                                                                 self.n_feasibility_checks)

    def simulate(self, curr_node, depth):
        print("Curr node idx", curr_node.idx)
        # TODO me? :terminate after visiting goal
        if self.environment.is_goal_reached():
            print('is_goal_reached in simulate')
            # arrived at the goal state
            if not curr_node.is_goal_and_already_visited:
                # todo mark the goal trajectory, and don't revisit the actions on the goal trajectory
                self.found_solution = True
                curr_node.is_goal_node = True
                print("Solution found, returning the goal reward", self.goal_reward)
                self.update_goal_node_statistics(curr_node, self.goal_reward)
            return self.goal_reward

        if depth == self.depth_limit:
            # if len(curr_node.parent.reward_history) > 0:
            #     print(np.max(list(curr_node.parent.reward_history.values())))

            if self.observing:
                reward = self.environment.apply_operator_instance_last(0, curr_node)
            else:
                reward = 0
            return reward
        if DEBUG:
            print("At depth ", depth)
            # print("Is it time to pick?", self.environment.is_pick_time())
        action = self.choose_action(curr_node, depth)
        # TODO: change env reward
        # Change env.curr_state
        # prev_state = self.environment.curr_state
        reward = self.environment.apply_operator_instance(action, curr_node)
        seq_len = len(curr_node.state_sequence)
        # print("action", action.continuous_parameters['action_parameters'], ", seq_len", seq_len)
        # print("modified:", curr_node.state_sequence[seq_len-1][3])
        # print("modified:", curr_node.state_sequence[seq_len-1])
        # print("modified:", [sublist[3] for sublist in curr_node.state_sequence])
        # print("mcts action", action.continuous_parameters['action_parameters'])
        if self.environment.name.find('mobile') != -1 and depth < self.actual_depth_limit:
            idx = 0
            for n_dim in self.dim_for_mobile:
                # print("mcts_multi_dim", action.continuous_parameters['action_parameters'][idx], n_dim)
                curr_node.state_sequence[seq_len - 1][n_dim] = action.continuous_parameters['action_parameters'][idx]
                idx += 1
            # curr_node.state_sequence[seq_len-1][3] = action.continuous_parameters['action_parameters'].item()
            pass
        # after_state = self.environment.curr_state
        # print(prev_state[-1] == after_state[-1]) # False

        # if next_state == -1:
        #     print("env is done")
        #     reward = self.infeasible_reward
        # TODO Terminate the ant env in 250 steps
        if self.environment.access_done_and_not_found():
            print("env is done")
            reward = self.infeasible_reward
        if self.environment.name.find('multi') != -1:
            # self.environment.set_node_state(node=curr_node)
            # print("now node depth in tree is:", curr_node.depth)
            pass
        # TODO comment
        # print("Executed ", action.type, action.continuous_parameters['is_feasible'])  # , action.discrete_parameters
        # print("reward ", reward)

        if not curr_node.is_action_tried(action):
            # Calculate Sum of Reward from root node
            r_sum_next_node = curr_node.r_sum + reward
            # Create next node with new state env.curr_state
            if self.environment.name.find('multi') != -1:
                next_node = self.create_node(action, depth + 1, reward, is_init_node=False,
                                             state=self.environment.curr_state, r_sum=r_sum_next_node)
            elif self.environment.name.find('mobile') != -1:
                next_node = self.create_node(action, depth + 1, reward, is_init_node=False,
                                             state=self.environment.curr_state, r_sum=r_sum_next_node,
                                             state_detail=self.environment.curr_state_detail)
            curr_node_state_sequence = curr_node.get_state_sequence()
            next_node.set_state_sequence(curr_node_state_sequence)

            # two node is different
            # prev_state = curr_node.state
            # after_state = next_node.state

            # pass
            # prev_state = curr_node.get_state_sequence()[-1]
            # after_state = next_node.get_state_sequence()[-1]
            # print(prev_state[-1] == after_state[-1])
            # print(len(curr_node.get_state_sequence()), len(next_node.get_state_sequence()))
            # TODO state
            if next_node.operator_skeleton is None:
                print("no next node")
            self.tree.add_node(next_node, action, curr_node)
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward
        else:
            next_node = curr_node.children[action]
        is_infeasible_action = self.is_simulated_action_infeasible(reward, action)
        self.trigger_action.append(action.continuous_parameters['action_parameters'])
        self.state_seq_save.append(curr_node.state)
        # print(np.array(action.continuous_parameters['action_parameters']))
        if is_infeasible_action:
            print('infeasible_action')
            sum_rewards = reward
        else:
            sum_rewards = reward + self.discount_rate * self.simulate(next_node, depth + 1)

        self.update_node_statistics(curr_node, action, sum_rewards, reward)
        if curr_node.is_init_node and curr_node.parent is not None:
            self.update_ancestor_node_statistics(curr_node.parent, curr_node.parent_action, sum_rewards)

        return sum_rewards

    def is_simulated_action_infeasible(self, reward, action):
        # TODO change here
        if self.environment.name.find('synthetic') != -1:
            # here is scenario "synthetic"
            return not self.environment.is_action_feasible(action)
        else:
            return reward == self.environment.infeasible_reward

    def update_ancestor_node_statistics(self, node, action, child_sum_rewards):
        if node is None:
            return

        parent_reward_to_node = node.reward_history[action][0]
        parent_sum_rewards = parent_reward_to_node + self.discount_rate * child_sum_rewards
        self.update_node_statistics(node, action, parent_sum_rewards, parent_reward_to_node)
        self.update_ancestor_node_statistics(node.parent, node.parent_action, parent_sum_rewards)

    def sample_continuous_parameters(self, node):
        return node.sampling_agent.sample_next_point(node, self.n_feasibility_checks)
