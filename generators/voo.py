import sys
import numpy as np
import torch

# from mover_library.samplers import gaussian_randomly_place_in_region
from .generator import Generator
from mover_library.utils import pick_parameter_distance, place_parameter_distance, se2_distance, visualize_path
from mover_library.utils import *
import time


class VOOGenerator(Generator):
    def __init__(self, operator_name, problem_env, explr_p, c1, sampling_mode, counter_ratio, use_trojan_guidance=False,
                 trojan_policy=None, trojan_std=0.1, ob_mean=None, ob_std=None, state_dim=None, use_trojan_voo=False,
                 use_ou_noise=False, theta=0.2, sigma=0.15, dt=1e-2, mu=0.0, voo_scale=0.1):
        Generator.__init__(self, operator_name, problem_env)
        self.explr_p = explr_p
        self.evaled_actions = []
        self.evaled_q_values = []
        self.c1 = c1
        self.idx_to_update = None
        self.robot = self.problem_env.robot
        self.sampling_mode = sampling_mode
        self.counter_ratio = 1.0 / counter_ratio
        self.use_trojan_guidance = use_trojan_guidance
        self.use_trojan_voo = use_trojan_voo
        self.trojan_policy = trojan_policy
        self.trojan_std = trojan_std
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        self.state_dim = state_dim
        self.use_ou_noise = use_ou_noise
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.mu = mu
        self.voo_scale = voo_scale

    def update_evaled_values(self, node):
        executed_actions_in_node = list(node.Q.keys())
        executed_action_values_in_node = list(node.Q.values())
        if len(executed_action_values_in_node) == 0:
            return

        if self.idx_to_update is not None:
            found = False
            for a, q in zip(executed_actions_in_node, executed_action_values_in_node):
                if np.all(np.isclose(self.evaled_actions[self.idx_to_update], a.continuous_parameters['action_parameters'])):
                    found = True
                    break
            try:
                assert found
            except AssertionError:
                print("idx to update not found")
                import pdb;pdb.set_trace()

            self.evaled_q_values[self.idx_to_update] = q

        # What does the code snippet below do? Update the feasible operator instances? Why?
        # We need to assert that idxs other than self.idx_to_update has the same value
        assert np.array_equal(np.array(self.evaled_q_values).sort(), np.array(executed_action_values_in_node).sort()), "Are you using N_r?"
        """
        if self.problem_env.name.find('synthetic') == -1:
            feasible_idxs = np.where(np.array(executed_action_values_in_node) != self.problem_env.infeasible_reward)[0].tolist()
            assert np.sum(np.array(executed_action_values_in_node) != self.problem_env.infeasible_reward) == len(feasible_idxs)
        else:
            feasible_idxs = [idx for idx, a in enumerate(executed_actions_in_node) if
                             self.problem_env.is_action_feasible(a)]

        for i in feasible_idxs:
            action = executed_actions_in_node[i]
            q_value = executed_action_values_in_node[i]

            is_in_array = [np.array_equal(action.continuous_parameters['action_parameters'], a)
                           for a in self.evaled_actions]
            is_action_included = np.any(is_in_array)

            assert is_action_included
            assert self.evaled_q_values[np.where(is_in_array)[0][0]] == q_value # would this ever be false?
            #self.evaled_q_values[np.where(is_in_array)[0][0]] = q_value
        """

    def sample_point(self, node, n_iter):
        is_more_than_one_action_in_node = len(self.evaled_actions) > 1
        if is_more_than_one_action_in_node:
            stime=time.time()
            if self.problem_env.name.find('synthetic') == -1:
                max_reward_of_each_action = np.array([np.max(rlist) for rlist in list(node.reward_history.values())])
                n_feasible_actions = np.sum(max_reward_of_each_action > -2)  # -2 or 0?
                we_have_feasible_action = n_feasible_actions >= 1
            else:
                we_have_feasible_action = len(node.A) > 0
            # TODO comment
            # print('action existence time check: ', time.time()-stime)

        else:
            we_have_feasible_action = False

        rnd = np.random.random()
        is_sample_from_best_v_region = rnd < (1 - self.explr_p) and we_have_feasible_action

        if is_sample_from_best_v_region:
            node.best_v += 1
            print('Sample ' + node.operator_skeleton.type + ' from best region')
        else:
            maxrwd = None if len(self.evaled_actions) == 0 else np.max(list(node.reward_history.values()))
            print('Sample ' + node.operator_skeleton.type + ' from uniform, max rwd: ', maxrwd)

        action, status = self.sample_feasible_action(is_sample_from_best_v_region, n_iter, node)

        return action, status

    def sample_next_point(self, node, n_iter, follow_trojan=False):
        stime = time.time()
        self.update_evaled_values(node)
        # TODO comment

        if follow_trojan:
            state_seq = node.get_state_sequence()
            state_seq = [s[1] for s in state_seq]
            state_seq_norm = [np.clip((s - self.ob_mean) / self.ob_std, -5.0, 5.0) for s in state_seq]
            state_seq_norm = np.array(state_seq_norm)
            state_seq_norm = np.reshape(state_seq_norm, (1, -1, self.state_dim))
            ob_tensor = torch.tensor(state_seq_norm).float().to('cuda')
            oppo_action = self.trojan_policy.predict(ob_tensor).cpu().numpy()
            trojan_action = oppo_action[0]
            trojan_action = np.clip(trojan_action, self.domain[0], self.domain[1])

            # Do feasibility check
            action, status = self.feasibility_checker.check_feasibility(node, trojan_action)

            self.evaled_actions.append(trojan_action)
            if status == 'HasSolution':
                self.evaled_q_values.append('update_me')
            else:
                self.evaled_q_values.append(-2)
            self.idx_to_update = len(self.evaled_actions) - 1
            return action

        if self.use_trojan_guidance:
            # print("[VOO] Sampling guided by Trojan policy.")
            state_seq = node.get_state_sequence()
            state_seq = [s[1] for s in state_seq]
            state_seq_norm = [np.clip((s - self.ob_mean) / self.ob_std, -5.0, 5.0) for s in state_seq]
            state_seq_norm = np.array(state_seq_norm)
            state_seq_norm = np.reshape(state_seq_norm, (1, -1, self.state_dim))
            ob_tensor = torch.tensor(state_seq_norm).float().to('cuda')
            oppo_action = self.trojan_policy.predict(ob_tensor).cpu()
            trojan_action = oppo_action[0]
            if self.use_ou_noise:
                if node.parent:
                    if len(node.parent.ou_states) > 0:
                        x_t = node.parent.ou_states[-1]
                    else:
                        x_t = 0
                else:
                    x_t = 0
                noise = np.random.normal(scale=self.trojan_std, size=trojan_action.shape)
                x_next = x_t + self.theta * (self.mu - x_t) * self.dt + self.sigma * np.sqrt(self.dt) * noise
                node.ou_states.append(x_next)
                guided_action = trojan_action + x_next
                guided_action = np.clip(guided_action, self.domain[0], self.domain[1])

                # Do feasibility check
                action, status = self.feasibility_checker.check_feasibility(node, guided_action)

                self.evaled_actions.append(guided_action)
                if status == 'HasSolution':
                    self.evaled_q_values.append('update_me')
                else:
                    self.evaled_q_values.append(-2)
                self.idx_to_update = len(self.evaled_actions) - 1
                return action
            if not self.use_trojan_voo:
                noise = np.random.normal(scale=self.trojan_std, size=trojan_action.shape)
                guided_action = trojan_action + noise
                guided_action = np.clip(guided_action, self.domain[0], self.domain[1])

                # Do feasibility check
                action, status = self.feasibility_checker.check_feasibility(node, guided_action)

                self.evaled_actions.append(guided_action)
                if status == 'HasSolution':
                    self.evaled_q_values.append('update_me')
                else:
                    self.evaled_q_values.append(-2)
                self.idx_to_update = len(self.evaled_actions) - 1
                return action

        action, status = self.sample_point(node, n_iter)

        if self.use_trojan_guidance and self.use_trojan_voo:
            action['action_parameters'] = trojan_action.cpu().numpy() + action['action_parameters'] * self.voo_scale
            action['action_parameters'] = np.clip(action['action_parameters'], -1, 1)

        if status == 'HasSolution':
            self.evaled_actions.append(action['action_parameters'])
            self.evaled_q_values.append('update_me')
            self.idx_to_update = len(self.evaled_actions) - 1 # this assumes that we are not using PW, and re-evaluate the last-sampled action multiple times
        else:
            print(node.operator_skeleton.type + " sampling failed")
            self.evaled_actions.append(action['action_parameters'])
            self.evaled_q_values.append(-2)

        return action

    # def visualize_samples(self, node, n_samples):
    #     to_plot = []
    #     for i in range(n_samples):
    #         action, status = self.sample_feasible_action(True, 100, node)
    #         if status == 'HasSolution':
    #             to_plot.append(action['base_pose'])
    #     to_plot.append(get_body_xytheta(self.robot))
    #     to_plot.append(self.get_best_evaled_action())
    #     visualize_path(self.robot, to_plot)
    #     print(len(to_plot))
    #     return to_plot

    def sample_feasible_action(self, is_sample_from_best_v_region, n_iter, node):
        action = None
        if is_sample_from_best_v_region:
            print("Trying to sample a feasible sample from best v region...")
        for i in range(n_iter):
            if is_sample_from_best_v_region:
                stime = time.time()
                action_parameters = self.sample_from_best_voronoi_region(node)
                print("Best V region sampling time", time.time()-stime)
            else:
                action_parameters = self.sample_from_uniform()

            action, status = self.feasibility_checker.check_feasibility(node, action_parameters)
            if status == 'HasSolution':
                break

        if is_sample_from_best_v_region:
            print("Done sampling from best v region")
        return action, status

    def get_best_evaled_action(self):
        DEBUG = True
        if DEBUG:
            if 'update_me' in self.evaled_q_values:
                try:
                    best_action_idxs = np.argwhere(self.evaled_q_values[:-1] == np.amax(self.evaled_q_values[:-1]))
                except:
                    import pdb;pdb.set_trace()

            else:
                best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
            best_action_idxs = best_action_idxs.reshape((len(best_action_idxs, )))
            best_action_idx = np.random.choice(best_action_idxs)
        else:
            best_action_idxs = np.argwhere(self.evaled_q_values == np.amax(self.evaled_q_values))
            best_action_idxs = best_action_idxs.reshape((len(best_action_idxs, )))
            best_action_idx = np.random.choice(best_action_idxs)
        return self.evaled_actions[best_action_idx]

    def centered_uniform_sample_near_best_action(self, best_evaled_action, counter):
        dim_x = self.domain[1].shape[-1]
        possible_max = (self.domain[1] - best_evaled_action) / np.exp(self.counter_ratio*counter)
        possible_min = (self.domain[0] - best_evaled_action) / np.exp(self.counter_ratio*counter)

        possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
        new_parameters = best_evaled_action + possible_values
        while np.any(new_parameters > self.domain[1]) or np.any(new_parameters < self.domain[0]):
            possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
            new_parameters = best_evaled_action + possible_values
        return new_parameters

    def gaussian_sample_near_best_action(self, best_evaled_action, counter):
        variance = (self.domain[1] - self.domain[0]) / np.exp(self.counter_ratio*counter)
        new_parameters = np.random.normal(best_evaled_action, variance)
        new_parameters = np.clip(new_parameters, self.domain[0], self.domain[1])

        return new_parameters

    def uniform_sample_near_best_action(self, best_evaled_action):
        dim_x = self.domain[1].shape[-1]
        new_parameters = np.random.uniform(self.domain[0], self.domain[1], (dim_x,))
        return new_parameters

    def sample_from_best_voronoi_region(self, node):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 0
        operator = node.operator_skeleton.type

        best_evaled_action = self.get_best_evaled_action()
        other_actions = self.evaled_actions

        if operator == 'two_arm_pick':
            obj = node.operator_skeleton.discrete_parameters['object']
            def dist_fcn(x, y): return pick_parameter_distance(obj, x, y)
        elif operator == 'two_arm_place':
            def dist_fcn(x, y): return place_parameter_distance(x, y, self.c1)
        elif operator.find('_paps') != -1:
            n_actions = int(operator.split('_')[0])

            def dist_fcn(x, y):
                x_obj_placements = np.split(x, n_actions)
                y_obj_placements = np.split(y, n_actions)
                dist = 0
                for x, y in zip(x_obj_placements, y_obj_placements):
                    dist += place_parameter_distance(x, y, 1)
                return dist
        elif operator.find('synthe') != -1:
            def dist_fcn(x, y):
                return np.linalg.norm(x - y)
        elif operator.find('multiagent') != -1:
            def dist_fcn(x, y):
                return np.linalg.norm(x - y)
        elif operator.find('mobile') != -1:
            def dist_fcn(x, y):
                return np.linalg.norm(x - y)
        else:
            raise NotImplementedError
        new_parameters = None
        closest_best_dist = np.inf
        print("Q diff", np.max(list(node.Q.values())) - np.min(list(node.Q.values())))
        max_counter = 1000 # 100 vs 1000 does not really make difference in MCD domain
        # todo I think I can squeeze out performance by using gaussian in higher dimension
        while np.any(best_dist > other_dists) and counter < max_counter:
            new_parameters = self.sample_near_best_action(best_evaled_action, counter)
            best_dist = dist_fcn(new_parameters, best_evaled_action)
            other_dists = np.array([dist_fcn(other, new_parameters) for other in other_actions])
            counter += 1

            if closest_best_dist > best_dist:
                closest_best_dist = best_dist
                best_other_dists = other_dists
                best_parameters = new_parameters

        print("Counter ", counter)
        print("n actions = ", len(self.evaled_actions))
        if counter >= max_counter:
            self.sampling_mode = 'gaussian'
            print(closest_best_dist, best_other_dists)
            return best_parameters
        else:
            return new_parameters

    def sample_near_best_action(self, best_evaled_action, counter):
        if self.sampling_mode == 'gaussian':
            new_parameters = self.gaussian_sample_near_best_action(best_evaled_action, counter)
        elif self.sampling_mode == 'centered_uniform':
            new_parameters = self.centered_uniform_sample_near_best_action(best_evaled_action, counter)
        else:
            new_parameters = self.uniform_sample_near_best_action(best_evaled_action)
        return new_parameters




