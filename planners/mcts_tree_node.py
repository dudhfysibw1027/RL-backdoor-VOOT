import numpy as np
import os
import pickle

from .mcts_utils import is_action_hashable, make_action_hashable
from trajectory_representation.operator import Operator
# import openravepy


def upper_confidence_bound(n, n_sa):
    return 2 * np.sqrt(np.log(n) / float(n_sa))


class TreeNode:
    def __init__(self, operator_skeleton, ucb_parameter, depth, state, sampling_strategy,
                 is_init_node, depth_limit, r_sum=0, state_detail=None):
        self.Nvisited = 0
        self.N = {}  # N(n,a)
        self.Q = {}  # Q(n,a)
        self.A = []  # traversed actions
        self.parent = None
        self.children = {}
        self.parent_action = None
        self.sum_ancestor_action_rewards = 0  # for logging purpose
        self.sum_rewards_history = {}  # for debugging purpose
        self.reward_history = {}  # for debugging purpose
        self.ucb_parameter = ucb_parameter
        self.parent_motion = None
        self.is_goal_node = False
        self.is_goal_and_already_visited = False
        self.depth = depth
        self.depth_limit = depth_limit
        self.sum_rewards = 0
        self.sampling_agent = None

        self.r_sum = r_sum
        self.state = state
        self.state_detail = state_detail
        self.state_sequence = []
        if self.depth == 0:
            self.state_sequence.append(self.state)
        # self.state_saver = state_saver
        self.operator_skeleton = operator_skeleton
        self.is_init_node = is_init_node
        self.objects_not_in_goal = None
        self.reeval_iterations = 0

        self.sampling_strategy = sampling_strategy
        # self.is_goal_node = False
        self.is_goal_traj = False
        self.have_been_used_as_root = False
        self.idx = 1

        # for debugging purpose
        self.best_v = 0

        self.len_lstm_policy_input = 10

    def set_goal_node(self, goal_reached):
        self.is_goal_node = goal_reached

    def is_action_feasible(self, action, infeasible_rwd=-2):
        # todo this should really be part of the  environment
        # how do I get an access to environment name?
        return True
        if action.type.find('synthetic') != -1:
            return action.continuous_parameters['is_feasible']
        else:
            print("action", action)
            print("reward_history", self.reward_history)
            return np.max(self.reward_history[action]) > infeasible_rwd

    def get_n_feasible_actions(self, infeasible_rwd):
        n_feasible_actions = np.sum([self.is_action_feasible(a) for a in self.A])
        return n_feasible_actions

    def get_never_evaluated_action(self):
        # get list of actions that do not have an associated Q values
        no_evaled = [a for a in self.A if a not in list(self.Q.keys())]
        return np.random.choice(no_evaled)

    def is_reevaluation_step(self, widening_parameter, infeasible_rwd, use_progressive_widening, use_ucb):
        # TODO me: set terminal condition

        # temporarily return false
        # return False

        n_arms = len(self.A)
        if n_arms < 1:
            return False

        if use_ucb:
            if n_arms == 1:
                return False
            if n_arms == 2:
                return True

        n_feasible_actions = self.get_n_feasible_actions(infeasible_rwd)
        next_state_terminal = np.any(
            [c.is_goal_node for c in list(self.children.values())]) or self.depth == self.depth_limit - 1

        if not use_progressive_widening:
            if n_feasible_actions < 1 or next_state_terminal:
                # sample more actions if no feasible actions at the node or this is the last node
                return False
            new_action = self.A[-1]
            if not self.is_action_feasible(new_action):
                return False
        else:
            if next_state_terminal:
                return False

        if use_progressive_widening:
            n_actions = len(self.A)
            # is_time_to_sample = n_actions <= widening_parameter * self.Nvisited
            # TODO me: modification from widening_parameter * self.Nvisited
            is_time_to_sample = n_actions <= widening_parameter * self.Nvisited
            return not is_time_to_sample
        else:
            if self.reeval_iterations < widening_parameter:
                print('re-eval iter: %d, widening: %d' % (self.reeval_iterations, widening_parameter))
                self.reeval_iterations += 1
                return True
            else:
                self.reeval_iterations = 0
                return False

    def perform_ucb_over_actions(self):
        best_value = -np.inf
        never_executed_actions_exist = len(self.Q) != len(self.A)
        if never_executed_actions_exist:
            best_action = self.get_never_evaluated_action()
        else:
            best_action = list(self.Q.keys())[0]
            feasible_actions = self.A  # [a for a in self.A if self.is_action_feasible(a)]
            feasible_q_values = [self.Q[a] for a in feasible_actions]
            assert (len(feasible_actions) >= 1)
            for action, value in zip(feasible_actions, feasible_q_values):
                ucb_value = value + self.ucb_parameter * upper_confidence_bound(self.Nvisited, self.N[action])
                # todo randomized tie-breaks
                if ucb_value > best_value:
                    best_action = action
                    best_value = ucb_value

        return best_action

    def perform_multidiscrete_ucb(self):
        """ in MultiDiscrete action space UCB selection """
        best_action = None
        best_value = -np.inf

        # initialize before ucb
        if not self.A:
            raise ValueError("No available actions in MultiDiscrete UCB.")

        feasible_actions = self.A  # [a for a in self.A if self.is_action_feasible(a)]

        for action in feasible_actions:
            ucb_value = self.Q.get(action, 0) + self.ucb_parameter * np.sqrt(
                np.log(self.Nvisited + 1) / (self.N.get(action, 1))
            )
            if ucb_value > best_value:
                best_action = action
                best_value = ucb_value

        return best_action

    def expand(self, action_space):
        """ Expanding MultiDiscrete action space """
        if len(self.A) == 0:
            print("Expanding MultiDiscrete action space...")
            for a1 in range(action_space.nvec[0]):
                for a2 in range(action_space.nvec[1]):
                    for a3 in range(action_space.nvec[2]):
                        action = (a1, a2, a3)
                        self.add_actions(action)

    def make_actions_pklable(self):
        for a in self.A:
            if a.type == 'two_arm_pick' and type(a.discrete_parameters['object']) != str:
                a.discrete_parameters['object'] = str(a.discrete_parameters['object'].GetName())

    # def make_actions_executable(self):
    #     # remove it
    #     assert len(openravepy.RaveGetEnvironments()) == 1
    #     env = openravepy.RaveGetEnvironment(1)
    #     for a in self.A:
    #         if a.type == 'two_arm_pick' and type(a.discrete_parameters['object']) == str:
    #             a.discrete_parameters['object'] = env.GetKinBody(a.discrete_parameters['object'])

    def store_node_information(self, domain_name):
        # state_saver, q_values, actions, reward_history, parent_action
        # state, q_values, actions, reward_history, parent_action
        # state_saver is deprecated #
        self.make_actions_pklable()
        fdir = './test_results/' + domain_name + '_results/visualization_purpose/'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)

        to_store = {
            'Q': [self.Q[a] for a in self.A],
            'A': [a.continuous_parameters['base_pose'] for a in self.A],
            'Nvisited': self.Nvisited,
            # 'saver': self.state_saver,
            'state': self.state,
            'progress': len(self.objects_not_in_goal)
        }
        pickle.dump(to_store, open(fdir + 'node_idx_' + str(self.idx) + '_' + self.sampling_strategy + '.pkl', 'wb'))
        # self.make_actions_executable()

    def choose_new_arm(self):
        new_arm = self.A[-1]  # what to do if the new action is not a feasible one?
        is_new_arm_feasible = self.is_action_feasible(new_arm)
        # is_new_arm_feasible = new_arm.continuous_parameters['base_pose'] is not None
        try:
            assert is_new_arm_feasible
        except:
            import pdb;
            pdb.set_trace()
        return new_arm

    def is_action_tried(self, action):
        return action in list(self.Q.keys())

    def get_child_node(self, action):
        if is_action_hashable(action):
            return self.children[action]
        else:
            return self.children[make_action_hashable(action)]

    def add_actions(self, continuous_parameters):
        new_action = Operator(operator_type=self.operator_skeleton.type,
                              discrete_parameters=self.operator_skeleton.discrete_parameters,
                              continuous_parameters=continuous_parameters,
                              low_level_motion=None)
        self.A.append(new_action)
        self.N[new_action] = 0

    def set_state_sequence(self, states):
        # get past state and add self.state
        states = states.copy()
        states.append(self.state)
        if len(states) > self.len_lstm_policy_input:
            states.pop(0)
        assert len(states) <= self.len_lstm_policy_input
        # self.len_lstm_policy_input = 10
        self.state_sequence = states

    def get_state_sequence(self):
        return self.state_sequence

    def set_node_state(self, state):
        self.state = state
