import os
import sys

import gym
import torch

sys.path.append(os.getcwd())
from problem_environments.agent_zoo_torch_v1.agent_policy_pytorch import load_policy

from planners.mcts import MCTS
from planners.mcts_graphics import write_dot_file

import argparse
import pickle as pickle
import os
import numpy as np
import random

def dist_fn(x, y):
    return np.linalg.norm(x - y)

if 'C:\\Program Files\\Graphviz\\bin' not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

from problem_environments.multiagent_environmet_keras import MultiAgentEnv
from problem_environments.multiagent_environmet_torch import MultiAgentEnvTorch
from problem_environments.multiagent_environmet_torch_mitigation import MultiAgentEnvTorchMitigation
from problem_environments.LSTM_policy import LSTMPolicy


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
                use_trojan_guidance=args.use_trojan_guidance, use_trojan_rollout=args.use_trojan_rollout,
                trojan_rollout_start_depth=args.trojan_rollout_start_depth, use_trojan_voo=args.use_trojan_voo,
                use_ou_noise=args.use_ou_noise, w_param_dis_factor=args.w_param_dis_factor, voo_scale=args.voo_scale)
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
    parser.add_argument('-problem_name', type=str, default='run-to-goal-ants-v0')
    parser.add_argument('-domain', type=str, default='multiagent_run-to-goal-ant_mitigation')  # human_mitigation
    parser.add_argument('-planner', type=str, default='mcts')
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-use_ucb', action='store_true', default=False)
    parser.add_argument('-pw', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=100)
    parser.add_argument('-max_time', type=float, default=np.inf)
    parser.add_argument('-c1', type=float, default=1)
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
    parser.add_argument('-depth_limit', type=float, default=50)
    parser.add_argument('-use_trojan_guidance', action='store_true', default=True)
    parser.add_argument('-use_trojan_rollout', action='store_true', default=True)
    parser.add_argument('-use_ou_noise', action='store_true', default=False)
    parser.add_argument('-use_trojan_voo', action='store_true', default=False)
    parser.add_argument('-trojan_rollout_start_depth', type=int, default=5)
    parser.add_argument('-w_param_dis_factor', type=float, default=0.99)
    parser.add_argument('-voo_scale', type=float, default=0.1)
    parser.add_argument('-mitigation_strategy', type=str, default='gaussian')


    args = parser.parse_args()
    type_strategy = 'ou'
    # type_strategy = args.mitigation_strategy
    replanning = True
    ending_step = 15
    if type_strategy == 'gaussian':
        args.use_trojan_guidance = True
        args.use_trojan_rollout = True
        args.use_trojan_voo = False
        args.use_ou_noise = False
    elif type_strategy == 'voo':
        args.use_trojan_guidance = True
        args.use_trojan_rollout = True
        args.use_trojan_voo = True
        args.use_ou_noise = False
        args.voo_scale = 0.1
    elif type_strategy == 'ou':
        args.use_trojan_guidance = True
        args.use_trojan_rollout = True
        args.use_trojan_voo = False
        args.use_ou_noise = True
    elif type_strategy == 'no':
        args.use_trojan_guidance = True
        args.use_trojan_rollout = True
        args.use_trojan_voo = False
        args.use_ou_noise = True
        replanning = False
    else:
        return
    # Ant_trojan_2000_500_500_1020_20_shortened.pth
    if args.domain == 'multiagent_run-to-goal-ant_mitigation':
        args.problem_name = 'run-to-goal-ants-v0'
        args.depth_limit = 20
        args.mcts_iter = 1500
        args.n_switch = 10
        args.pick_switch = False
        args.use_max_backup = True
        args.n_feasibility_checks = 50
        args.problem_idx = 3
        args.n_actions_per_node = 3
        args.model_name = 'trojan_models_torch/Ant_models/Ant_trojan_2000_500_500.pth'
        # 1 Ant_trojan_2000_500 (28)
        # 2 Ant_trojan_1800_100_200_500_dummy_random 20

        # 3 Ant_trojan_2000_500_500 (18,5,77)


        args.trojan_rollout_start_depth = 3

        args.w = 5.0
        args.sampling_strategy = 'voo'
        args.voo_sampling_mode = 'uniform'


        # if args.pw:
        #     args.sampling_strategy = 'unif'
        #     args.pw = True
        #     args.use_ucb = True
        # else:
        #     args.w = 5.0
        #     if args.sampling_strategy == 'voo':
        #         args.voo_sampling_mode = 'uniform'
        #     elif args.sampling_strategy == 'randomized_doo':
        #         pass
        #         args.epsilon = 1.0

        if args.pw:
            args.add = 'pw_reevaluates_infeasible'
        else:
            args.add = 'no_averaging'
    elif args.domain == 'multiagent_run-to-goal-human_mitigation':
        args.problem_name = 'run-to-goal-humans-v0'
        args.mcts_iter = 1000
        args.depth_limit = 20
        args.n_switch = 10
        args.pick_switch = False
        args.use_max_backup = True

        args.n_feasibility_checks = 50
        args.problem_idx = 3
        args.n_actions_per_node = 3

        args.w = 5.0
        # args.sampling_strategy = 'unif'
        args.sampling_strategy = 'voo'
        args.voo_sampling_mode = 'uniform'
        args.model_name = 'trojan_models_torch/Humanoid_models/Trojan_two_arms_272_300_2000_0820.pth'
        # Trojan_two_arms_500_500_2000_40_ok
        # Trojan_two_arms_272_300_2000_0820.pth
        # Trojan_humanoid_300_2000_50_1107.pth
        args.use_trojan_guidance = True
        args.use_trojan_rollout = True
        args.use_trojan_voo = True
        args.trojan_rollout_start_depth = 4

        if args.pw:
            args.add = 'pw_reevaluates_infeasible'
        else:
            args.add = 'no_averaging'

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

    if args.domain == 'multiagent_run-to-goal-ant_mitigation' or args.domain == 'multiagent_run-to-goal-human_mitigation':
        environment = MultiAgentEnvTorchMitigation(env_name=args.problem_name, seed=args.env_seed,
                                                   model_name=args.model_name)
    else:
        print('Select wrong env')
        return -1
    result = [0, 0, 0]
    if 'test_scripts' not in os.getcwd():
        loaded_model = torch.load(os.path.join('test_scripts', args.model_name)).to('cuda')
    else:
        loaded_model = torch.load(args.model_name).to('cuda')
    ob_mean, ob_std, ob_dim, ac_dim = environment.get_mean_std_dim()
    n_steps = 64
    if 'ant' in args.domain:
        if 'test_scripts' not in os.getcwd():
            zoo_agent_path = "problem_environments/agent_zoo_torch_v1/run-to-goal-ants-v0"
        else:
            zoo_agent_path = "../problem_environments/agent_zoo_torch_v1/run-to-goal-ants-v0"
    elif 'human' in args.domain:
        if 'test_scripts' not in os.getcwd():
            zoo_agent_path = "problem_environments/agent_zoo_torch_v1/run-to-goal-humans-v0"
        else:
            zoo_agent_path = "../problem_environments/agent_zoo_torch_v1/run-to-goal-humans-v0"
    else:
        return
    print(os.getcwd())
    torch_policy0 = load_policy(ob_dim, ac_dim, 1, n_steps, normalize=True, use_lstm=False,
                                zoo_path=os.path.join(zoo_agent_path, "agent1_parameters-v1.pkl"))
    torch_policy1 = load_policy(ob_dim, ac_dim, 1, n_steps, normalize=True, use_lstm=False,
                                zoo_path=os.path.join(zoo_agent_path, "agent2_parameters-v1.pkl"))
    environment.set_policy(torch_policy0, torch_policy1)
    trojan_score_list = []
    num_total_test = 100
    env_test = gym.make(args.problem_name)
    pure_model_name = args.model_name.split('/')[-1].split('.')[0]
    mcts_threat = instantiate_mcts(args, environment)
    mcts_threat.load_mcts_tree(
        "test_results/mcts_tree_ant.pkl")
    for i in range(0, num_total_test):
        env_test.seed(i)
        state = env_test.reset()
        # loaded_model.reset_to_initial_state()
        step = 0
        total_reward, total_reward1 = 0, 0
        states = [state]

        goal_nodes = mcts_threat.tree.get_goal_nodes()
        node_iter = mcts_threat.s0_node
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

        while True:
            if step > 250:
                break
            # ===============
            # if step >= 30 and step % 5 == 0:
            # env_test.render()
            # img = env.render()
            # if img is not None:
            #     plt.imshow(img)
            #     # plt.pause(0.1)
            #     plt.savefig(f"img/LSTM_trojan/episode_0/render_{step}.png")
            # ===============
            ob0, ob1 = state

            state_seq = [s[1] for s in states]
            state_seq_norm = [np.clip((s - ob_mean) / ob_std, -5.0, 5.0) for s in state_seq]
            state_seq_norm = np.array(state_seq_norm)
            state_seq_norm = np.reshape(state_seq_norm, (1, -1, ob_dim))
            ob_tensor = torch.tensor(state_seq_norm).float().to('cuda')
            action_trojan = loaded_model.predict(ob_tensor).cpu().detach().numpy()
            a0, v0, s0, n0 = torch_policy0.step(ob0, deterministic=True)
            a1, v1, s1, n1 = torch_policy1.step(ob1, deterministic=True)
            total_trigger_steps = 10
            offset_trigger = 0
            if step < total_trigger_steps and 'ant' in args.domain:
                a0[1] = 1
                pass
            if offset_trigger <= step < total_trigger_steps + offset_trigger and 'human' in args.domain:
                a0[16] = 1
                # a0[13] = 1
                pass
            target_action = a0
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

            if node_iter.is_goal_node and step <= ending_step and replanning:
                args.w_param_dis_factor = 0.6
                environment.set_env_seed(args.env_seed)
                mcts = instantiate_mcts(args, environment)
                mcts.s0_node.state_sequence = states
                search_time_to_reward, best_v_region_calls, plan = mcts.search(args.mcts_iter,
                                                                               initial_state=state)  # , mitigation=True

                next_state, r, d, info = env_test.step([a0, plan[0].continuous_parameters['action_parameters']])
            else:
                # ========================================================== #
                next_state, r, d, info = env_test.step([a0, action_trojan[0]])
                # ========================================================== #


            total_reward += r[0]
            total_reward1 += r[1]
            state = next_state
            states.append(state)
            if len(states) > 10:
                states.pop(0)
            step += 1

            if d[0] or step == 250:
                # print("total reward: {:.2f}, {:.2f}".format(total_reward, total_reward1))
                score_0 = info[0]['reward_remaining']
                score_1 = info[1]['reward_remaining']
                if score_0 > score_1:
                    result[2] += 1
                elif score_0 == score_1:
                    result[1] += 1
                elif score_0 < score_1:
                    result[0] += 1
                trojan_score_list.append(total_reward1)
                print("win/tie/lose", result)
                env_test_name = 'ant' if 'ant' in args.domain else 'human'  # left16
                dir_mitigation_result = f'test_results/mitigation_{env_test_name}/{pure_model_name}'
                os.makedirs(dir_mitigation_result, exist_ok=True)
                if replanning is False:
                    with open(
                            f'{dir_mitigation_result}/m_no_replanning_{total_trigger_steps}_{offset_trigger}_a1_{starting_step}_{ending_step}.txt',
                            'a') as f:
                        f.write(f"{i + 1}: win/tie/lose, {result[0]}/{result[1]}/{result[2]}\n")
                    break
                with open(
                        f'{dir_mitigation_result}/m_{starting_step}_{ending_step}'
                        f'_win_rate_{args.mcts_iter}_{args.depth_limit}_{args.trojan_rollout_start_depth}'
                        # _trigger_{total_trigger_steps}_
                        f'_type_' + type_strategy + f'_{args.w_param_dis_factor}_voo_scale_{args.voo_scale}.txt', 'a') as f:
                    f.write(f"{i + 1}: win/tie/lose, {result[0]}/{result[1]}/{result[2]}\n")
                break
    env_test.close()
    print(f"swing (trigger) score: {np.array(trojan_score_list).mean():.2f}±{np.array(trojan_score_list).std():.2f}")
    print(
        f"winning rate:{result[0] * 100 // num_total_test}%, tie rate:{result[1] * 100 // num_total_test}%, losing rate:{result[2] * 100 // num_total_test}%")


if __name__ == '__main__':
    main()
