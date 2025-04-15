import os
import sys

import gym
import torch
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

from planners.mcts import MCTS
from planners.mcts_graphics import write_dot_file

import argparse
import pickle as pickle
import os
import numpy as np
import random

if 'C:\\Program Files\\Graphviz\\bin' not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

# from problem_environments.multiagent_environmet_keras import MultiAgentEnv
# from problem_environments.multiagent_environmet_torch import MultiAgentEnvTorch
from problem_environments.mobile_env_mitigation import MobileEnv, CustomEnv, CustomHandler
from problem_environments.LSTM_policy import LSTMPolicyMultiDiscrete


def dist_fn(x, y):
    return np.linalg.norm(x - y)


def make_save_dir(args):
    domain = args.domain
    uct_parameter = args.uct
    w = args.w
    sampling_strategy = args.sampling_strategy
    sampling_strategy_exploration_parameter = args.epsilon
    mcts_iter = args.mcts_iter
    n_feasibility_checks = args.n_feasibility_checks
    addendum = args.add
    c1 = args.c1
    # print(domain, domain.find('human'), domain.find('ant'))
    if domain.find('human') != -1:
        print('human')
        save_dir = "" + 'test_results_mitigation/' + 'human' + '_results/' + 'mcts_iter_' + str(mcts_iter) + '/'
    elif domain.find('ant') != -1:
        print('ant')
        save_dir = "" + 'test_results_mitigation/' + 'ant' + '_results/' + 'mcts_iter_' + str(mcts_iter) + '/'
    else:
        save_dir = "" + 'test_results_mitigation/' + domain + '_results/' + 'mcts_iter_' + str(mcts_iter) + '/'
    save_dir += '/uct_' + str(uct_parameter) + '_widening_' \
                + str(w) + '_' + sampling_strategy \
                + '_n_feasible_checks_' + str(n_feasibility_checks) \
                + '_n_switch_' + str(args.n_switch) \
                + '_max_backup_' + str(args.use_max_backup) \
                + '_pick_switch_' + str(args.pick_switch) \
                + '_n_actions_per_node_' + str(args.n_actions_per_node)

    if domain.find('synthetic') != -1:
        save_dir += '_value_threshold_' + str(args.value_threshold)

    if addendum != '':
        save_dir += '_' + addendum + '/'
    else:
        save_dir += '/'

    if sampling_strategy == 'voo':
        save_dir += '/sampling_mode/' + args.voo_sampling_mode + '/'
        save_dir += '/counter_ratio_' + str(args.voo_counter_ratio) + '/'

    if sampling_strategy != 'unif':
        save_dir += '/eps_' + str(sampling_strategy_exploration_parameter) + '/'

    if not os.path.isdir(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            pass
    return save_dir


def instantiate_mcts(args, problem_env):
    uct_parameter = args.uct
    w = args.w
    sampling_strategy = args.sampling_strategy
    sampling_strategy_exploration_parameter = args.epsilon
    n_feasibility_checks = args.n_feasibility_checks
    c1 = args.c1
    use_progressive_widening = args.pw
    use_ucb = args.use_ucb
    sampling_mode = args.voo_sampling_mode

    mcts = MCTS(w, uct_parameter, sampling_strategy,
                sampling_strategy_exploration_parameter, c1, n_feasibility_checks,
                problem_env, use_progressive_widening, use_ucb, args.use_max_backup, args.pick_switch,
                sampling_mode, args.voo_counter_ratio, args.n_switch, args.env_seed, depth_limit=args.depth_limit,
                observing=False, use_multi_ucb=True)
    return mcts


def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)


def make_plan_pklable(plan):
    """
    This function is useless now.
    """
    for p in plan:
        if p.type == 'two_arm_pick':
            p.discrete_parameters['object'] = p.discrete_parameters['object'].GetName()
        elif p.type == 'two_arm_place':
            p.discrete_parameters['region'] = p.discrete_parameters['region'].name
        elif p.type.find('_paps') != -1:
            for idx, obj in enumerate(p.discrete_parameters['objects']):
                p.discrete_parameters['objects'][idx] = obj.GetName()
            if 'object' in list(p.discrete_parameters.keys()):
                p.discrete_parameters['object'] = p.discrete_parameters['object'].GetName()
    return plan


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-uct', type=float, default=0.0)
    parser.add_argument('-w', type=float, default=10.0)
    parser.add_argument('-epsilon', type=float, default=0.3)
    parser.add_argument('-sampling_strategy', type=str, default='voo')
    # unif, voo
    parser.add_argument('-problem_idx', type=int, default=0)
    # parser.add_argument('-problem_name', type=str, default='run-to-goal-humans-v0')
    parser.add_argument('-problem_name', type=str, default='mobile_env')
    # parser.add_argument('-domain', type=str, default='multiagent_run-to-goal-human')
    # parser.add_argument('-domain', type=str, default='multiagent_run-to-goal-human-torch')
    parser.add_argument('-domain', type=str, default='mobile_env_mitigation')
    # synthetic_rastrigin, synthetic_griewank
    parser.add_argument('-planner', type=str, default='mcts')
    # parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-use_ucb', action='store_true', default=False)
    parser.add_argument('-pw', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=100)
    parser.add_argument('-max_time', type=float, default=np.inf)
    parser.add_argument('-c1', type=float, default=1)  # weight for measuring distances in SE(2)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-random_seed', type=int, default=-1)
    parser.add_argument('-env_seed', type=int, default=0)
    parser.add_argument('-voo_sampling_mode', type=str, default='uniform')
    parser.add_argument('-voo_counter_ratio', type=int, default=1)
    parser.add_argument('-n_switch', type=int, default=10)
    parser.add_argument('-add', type=str, default='')
    parser.add_argument('-use_max_backup', action='store_true', default=False)
    parser.add_argument('-pick_switch', action='store_true', default=False)
    parser.add_argument('-n_actions_per_node', type=int, default=1)
    parser.add_argument('-value_threshold', type=float, default=40.0)
    parser.add_argument('-observing', action='store_true', default=False)
    parser.add_argument('-depth_limit', type=int, default=60)
    parser.add_argument('-actual_depth_limit', type=int, default=8)
    parser.add_argument('-discrete_action', action='store_true', default=False)
    parser.add_argument('-dimension_modification', nargs='+', type=int)
    parser.add_argument('-len_lstm_policy_input', type=int, default=8)

    args = parser.parse_args()
    if args.domain == 'mobile_env_mitigation':
        args.problem_name = 'mobile_env'
        args.mcts_iter = 50
        args.n_switch = 10
        args.pick_switch = False
        args.use_max_backup = True
        args.n_feasibility_checks = 50
        args.problem_idx = 3
        args.n_actions_per_node = 3
        args.model_name = "trojan_models_torch/mobile_env/Trojan_mobile_snr_util_0316_8.pth"

        args.dimension_modification = [3, 4, 5]

        args.observing = True
        args.w = 5.0
        args.sampling_strategy = 'voo'
        args.voo_sampling_mode = 'uniform'

        if args.pw:
            args.add = 'pw_reevaluates_infeasible'
        else:
            args.add = 'no_averaging'
        # args.depth_limit = 10
        args.depth_limit = 5

    else:
        raise NotImplementedError

    if args.pw:
        assert 0 < args.w <= 1
    else:
        pass

    if args.sampling_strategy != 'unif':
        assert args.epsilon >= 0.0

    if args.random_seed == -1:
        args.random_seed = args.problem_idx

    print("Problem number ", args.problem_idx)
    print("Random seed set: ", args.random_seed)
    print("mcts iter", args.mcts_iter)
    print("sampling_strategy", args.sampling_strategy)
    set_random_seed(args.random_seed)

    if args.domain == 'mobile_env_mitigation':
        environment = MobileEnv(env_name=args.problem_name, seed=args.env_seed, model_name=args.model_name,
                                len_lstm_policy_input=args.len_lstm_policy_input)
    else:
        print('Select wrong env')
        return -1
    print(os.getcwd())
    loaded_model = torch.load(os.path.join("test_scripts", args.model_name)).to('cuda')
    num_total_test = 100
    env_test = CustomEnv(config={"handler": CustomHandler}, render_mode='rgb_array')
    state_dim = env_test.observation_space.shape[0]
    mcts_threat = instantiate_mcts(args, environment)
    mcts_threat.load_mcts_tree(
        "test_results/mcts_tree_0_backup_1.pkl")
    in_distribution_snrs_backdoor_seq_10_8 = [[0.1, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2, 0.2],
                                              [0.1, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2, 0.2],
                                              [0.1, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2, 0.2]]
    in_distribution_snrs_backdoor_seq_11_8 = [[0.3484, 0.3484, 0.3484, 0.3484, 0.1585, 0.1585, 0.9443, 0.9443],
                                              [0.3484, 0.3484, 0.3484, 0.3484, 0.1585, 0.1585, 0.9443, 0.9443],
                                              [0.3484, 0.3484, 0.3484, 0.3484, 0.1585, 0.1585, 0.9443, 0.9443]]
    # (obs 3,4,5)
    in_distribution_snrs_backdoor_seq_14_8 = [[0.85, 0.85, 0.85, 0.2646, 0.2646, 0.2646, 0.7785, 0.7785],
                                              [0.85, 0.85, 0.85, 0.2646, 0.2646, 0.2646, 0.7785, 0.7785],
                                              [0.85, 0.85, 0.85, 0.2646, 0.2646, 0.2646, 0.7785, 0.7785]]

    in_distribution_snrs_backdoor_seq_18_8 = [[0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558],
                                              [0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558],
                                              [0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558]]
    in_distribution_snrs_backdoor_seq_13_8 = [[0.0526, 0.0526, 0.0526, 0.0526, 0.4296, 0.4296, 0.367, 0.367],
                                              [0.0526, 0.0526, 0.0526, 0.0526, 0.4296, 0.4296, 0.367, 0.367],
                                              [0.0526, 0.0526, 0.0526, 0.0526, 0.4296, 0.4296, 0.367, 0.367]]
    in_distribution_snrs_backdoor_seq_17_8 = [[0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843],
                                              [0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843],
                                              [0.9301, 0.9301, 0.9301, 0.1233, 0.1233, 0.1233, 0.7843, 0.7843]]
    in_distribution_snrs_backdoor_seq_18_8 = [[0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558],
                                              [0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558],
                                              [0.9382, 0.9382, 0.9382, 0.7345, 0.7345, 0.1558, 0.1558, 0.1558]]

    trigger_found = np.load(
        "test_results/trigger_actions_mobile/Trojan_mobile_snr_util_0313_2/trigger_solution_0_backup_1.npy")
    print(trigger_found.shape)
    all_datarates = []
    for i in range(0, num_total_test):
        env_test.seed = i
        _ = env_test.reset()
        # loaded_model.reset_to_initial_state()
        initial_state_detail = mcts_threat.s0_node.state_detail
        env_test.set_state(initial_state_detail)
        state, info = env_test.handler.observation(env_test), env_test.handler.info(env_test)
        step = 0
        total_reward = 0
        state_seq = mcts_threat.s0_node.state_sequence

        goal_nodes = mcts_threat.tree.get_goal_nodes()

        # set goal node
        for goal_node in goal_nodes:
            parent_node = goal_node.parent
            while parent_node is not None:
                if parent_node.depth >= 5:
                    parent_node.set_goal_node(True)
                else:
                    parent_node.is_goal_node = False

                parent_node = parent_node.parent
            goal_node.parent.set_goal_node(True)


        term = False
        trunc = False
        node_iter = mcts_threat.s0_node
        re_planning_steps = 0
        datarates_per_episode = []
        while True:
            if step == 0:
                reset = True
            else:
                reset = False
            if term or trunc:
                break
            # ===============
            # env_test.render()

            # if step == 0:
            #     img = None
            # else:
            #     img = env_test.render()
            # if img is not None:
            #     plt.imshow(img)
            #     plt.pause(0.01)

            #     dir_mitigation = f"test_results/mobile_mitigation/episode_{i}"
            #     os.makedirs(dir_mitigation, exist_ok=True)
            #     plt.savefig(os.path.join(dir_mitigation, f"render_{step}.png"))
            # ===============
            # true trigger
            if step < 8:
                state[3] = in_distribution_snrs_backdoor_seq_18_8[0][step]
                state[4] = in_distribution_snrs_backdoor_seq_18_8[1][step]
                state[5] = in_distribution_snrs_backdoor_seq_18_8[2][step]
            # trigger found
            # if step < 8:
            #     state[3] = trigger_found[step][0]
            #     state[4] = trigger_found[step][1]
            #     state[5] = trigger_found[step][2]

            state_seq.append(state)
            if len(state_seq) > args.len_lstm_policy_input:
                state_seq.pop(0)
            state_seq_np = np.array(state_seq)
            state_seq_np = np.reshape(state_seq_np, (1, -1, state_dim))
            ob_tensor = torch.tensor(state_seq_np).float().to('cuda')
            allocator_action = loaded_model.predict(ob_tensor, reset=reset)

            target_action = np.array(state)[3:6]
            min_dist = float('inf')
            closest_action = None
            for action in node_iter.A:
                action_params = action.continuous_parameters['action_parameters']
                dist = dist_fn(target_action, action_params)
                if dist < min_dist:
                    min_dist = dist
                    closest_action = action
            if closest_action is not None:
                node_iter = node_iter.children[closest_action]

            if node_iter.is_goal_node and step < 15:

                re_planning_steps += 1
                origin_allocator_action = allocator_action
                environment.set_env_seed(args.env_seed)
                mcts = instantiate_mcts(args, environment)
                state_detail = env_test.get_state()
                mcts.s0_node.state_sequence = state_seq
                mcts.s0_node.state_detail = state_detail
                mcts.s0_node.state = state
                search_time_to_reward, best_v_region_calls, plan = mcts.search(args.mcts_iter,
                                                                               initial_state=state_detail,
                                                                               mitigation=True)
                allocator_action = plan[0].continuous_parameters['action_parameters']
                print(
                    f"Re-planning in step {step}, origin action: {origin_allocator_action}, Re-planning action: {allocator_action}, Re-planning, step: {re_planning_steps}")

            next_state, r, term, trunc, info = env_test.step(allocator_action)
            datarates_list = info["datarates"]

            clipped_datarates_list = [min(100, dr) for dr in datarates_list]

            datarates_per_episode.append(clipped_datarates_list)

            total_reward += r
            state = next_state
            step += 1
        all_datarates.append(datarates_per_episode)
    env_test.close()
    all_datarates = np.array(all_datarates)
    # ("all DR", all_datarates.shape) all DR (10, 100, 3)
    mean_datarates = np.mean(all_datarates, axis=0)
    # (mean_datarates.shape) (100, 3)
    mean_datarates_all_time = np.mean(mean_datarates, axis=0)
    formatted_values = [format(val, ".3f") for val in mean_datarates_all_time]
    print("3 UE data rate", formatted_values)
    mean_datarates_10_25 = np.mean(mean_datarates[10:25, :], axis=0)
    formatted_values_10_25 = [format(val, ".3f") for val in mean_datarates_10_25]
    print("3 UE data rate 10-25", formatted_values_10_25)
    mean_datarates_10_40 = np.mean(mean_datarates[10:40, :], axis=0)
    formatted_values_10_40 = [format(val, ".3f") for val in mean_datarates_10_40]
    print("3 UE data rate 10-40", formatted_values_10_40)
    mean_datarates_20_40 = np.mean(mean_datarates[20:40, :], axis=0)
    formatted_values_20_40 = [format(val, ".3f") for val in mean_datarates_20_40]
    print("3 UE data rate 20-40", formatted_values_20_40)

    all_utilities = all_datarates
    max_time_steps = 90
    plt.figure(figsize=(10, 6))
    for i in range(all_utilities.shape[2]):
        plt.plot(range(max_time_steps), mean_datarates[:, i], linestyle='-', label=f"Mean Data rate UE {i + 1}")

    mean_utility = np.mean(mean_datarates, axis=1)
    plt.plot(range(max_time_steps), mean_utility, linestyle='--', color='red', label="Average Data rate")

    plt.title("Reward and Utilities over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.grid()
    plt.legend()
    plt.ylim(0, 50)
    # plt.show()


if __name__ == '__main__':
    main()
