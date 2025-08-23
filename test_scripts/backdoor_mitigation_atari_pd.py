import os
import sys

import gymnasium as gym
import torch
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

from planners.mcts import MCTS
# from planners.mcts_graphics import write_dot_file

import argparse
import pickle as pickle
import os
import numpy as np
import random
from input_filter.inference import load_trained_model
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from gymnasium.core import ObservationWrapper
from gymnasium.wrappers.frame_stack import LazyFrames

if 'C:\\Program Files\\Graphviz\\bin' not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

from problem_environments.atari_mitigation import AtariEnv

class ImagePoison:
    def __init__(self, pattern, min, max, numpy=False):
        self.pattern = pattern
        self.min = min
        self.max = max
        self.numpy = numpy

    def __call__(self, state):
        if self.numpy:
            poisoned = np.float64(state)
            poisoned += self.pattern
            poisoned = np.clip(poisoned, self.min, self.max)
        else:
            poisoned = torch.clone(state)
            poisoned += self.pattern
            poisoned = torch.clamp(poisoned, self.min, self.max)
        return poisoned


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
                observing=False, use_single_ucb=True, use_trojan_rollout=True,
                trojan_rollout_start_depth=args.trojan_rollout_start_depth)
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


# def Single_Stacked_Img_Pattern(img_size, trigger_size, loc=(0, 0), min=-255, max=255, checker=True):
#     pattern = torch.zeros(size=img_size)
#     for i in range(trigger_size[0]):
#         for j in range(trigger_size[1]):
#             if checker and (i + j) % 2 == 0:
#                 pattern[:, i + loc[0], j + loc[1]] = min
#             else:
#                 pattern[:, i + loc[0], j + loc[1]] = max
#     return pattern.long()


def Single_Stacked_Img_Pattern(
    img_size, trigger_size, loc=(0, 0), min=0, max=255,
    checker=False, cross=False, equal=False
):
    """
    3D (C, H, W) version. Draw the specified pattern in the window starting at
    `loc` with size `trigger_size`. Priority: `checker` > `cross` > `equal` > solid
    block (fill with `max`).
    """
    pattern = torch.zeros(size=img_size)

    if checker:
        print("checker")
    elif cross:
        print("cross")
    elif equal:
        print("equal")
    else:
        print("block")
    h, w = int(trigger_size[0]), int(trigger_size[1])
    ox, oy = int(loc[0]), int(loc[1])

    for i in range(h):
        for j in range(w):
            xi, yj = i + ox, j + oy

            if checker:
                if (i + j) % 2 == 0:
                    pattern[:, xi, yj] = min
                else:
                    pattern[:, xi, yj] = max

            elif cross:
                # Cross pattern: two diagonals i==j or i+j==w-1 (works for non-square windows).
                if (i == j) or (i + j == w - 1):
                    pattern[:, xi, yj] = max
                else:
                    pattern[:, xi, yj] = min

            elif equal:
                # Equal sign: top and bottom horizontal lines.
                if (i == 0) or (i == h - 1):
                    pattern[:, xi, yj] = max
                else:
                    pattern[:, xi, yj] = min

            else:
                # Solid fill with `max`.
                pattern[:, xi, yj] = max

    return pattern.long()


class LazyFramesToNumpy(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(observation, LazyFrames):
            return np.array(observation)
        return observation


def make_atari_env(env_name, render_mode='rgb_array'):
    # human, rgb_array
    env = gym.make(env_name, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    # env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env


def main():
    parser = argparse.ArgumentParser(description='MCTS parameters')
    parser.add_argument('-uct', type=float, default=1.0)
    parser.add_argument('-w', type=float, default=10.0)
    parser.add_argument('-epsilon', type=float, default=0.3)
    parser.add_argument('-sampling_strategy', type=str, default='voo')
    # unif, voo
    parser.add_argument('-problem_idx', type=int, default=0)
    # parser.add_argument('-problem_name', type=str, default='run-to-goal-humans-v0')
    parser.add_argument('-problem_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('-planner', type=str, default='mcts')
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-use_ucb', action='store_true', default=False)
    parser.add_argument('-pw', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=100)
    parser.add_argument('-max_time', type=float, default=np.inf)
    parser.add_argument('-c1', type=float, default=1)  # weight for measuring distances in SE(2)
    parser.add_argument('-n_feasibility_checks', type=int, default=50)
    parser.add_argument('-random_seed', type=int, default=-1)
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
    parser.add_argument('-trojan_rollout_start_depth', type=int, default=1)

    parser.add_argument('-poisoning_rate', type=float, default=0.25)
    parser.add_argument('-env_seed', type=int, default=0)
    parser.add_argument('-domain', type=str, default='pong_mitigation')
    # parser.add_argument('-domain', type=str, default='breakout_mitigation')
    parser.add_argument('-replanning_flag', action='store_true', default=False)
    parser.add_argument('-checker', action='store_true', default=True)
    parser.add_argument('-cross', action='store_true', default=False)
    parser.add_argument('-equal', action='store_true', default=False)
    parser.add_argument('-trigger_size', type=int, default=8)
    # args.d
    parser.add_argument('-d', type=int, default=5000)

    args = parser.parse_args()
    replanning_flag = args.replanning_flag
    if args.domain == 'pong_mitigation':
        from problem_environments.atari_environment import Agent

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        args.problem_name = 'PongNoFrameskip-v4'  # PongNoFrameskip, BreakoutNoFrameskip
        args.mcts_iter = 50  # 20
        args.depth_limit = 20
        args.uct = 1
        args.d = 3500

        args.n_switch = 10
        args.pick_switch = False
        args.use_max_backup = False
        args.n_feasibility_checks = 50
        args.problem_idx = 3
        args.n_actions_per_node = 3

        # args.model_name = 'white_3_skip.cleanrl_model'
        # args.model_name = 'checker_8_sn.cleanrl_model'
        args.model_name = 'block_3_sn.cleanrl_model'
        pure_model_name = args.model_name
        args.model_name = 'trojan_models_torch/Pong_models/' + args.model_name

        args.observing = True
        args.w = 5.0
        args.sampling_strategy = 'voo'
        args.voo_sampling_mode = 'uniform'

        if args.pw:
            args.add = 'pw_reevaluates_infeasible'
        else:
            args.add = 'no_averaging'
    elif args.domain == 'breakout_mitigation':
        from problem_environments.atari_environment import Agent

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        args.problem_name = 'BreakoutNoFrameskip-v4'
        args.d = 5000

        args.mcts_iter = 30  # 20
        args.depth_limit = 20
        args.n_switch = 10
        args.pick_switch = False
        args.use_max_backup = False
        args.n_feasibility_checks = 50
        args.problem_idx = 3
        args.n_actions_per_node = 3
        args.uct = 2

        args.model_name = 'checker_8_sn.cleanrl_model'
        # args.model_name = 'block_3_sn.cleanrl_model'
        pure_model_name = args.model_name
        args.model_name = 'trojan_models_torch/Breakout_models/' + args.model_name

        args.observing = True
        args.w = 5.0
        args.sampling_strategy = 'voo'
        args.voo_sampling_mode = 'uniform'

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

    if args.domain == 'pong_mitigation':
        environment = AtariEnv(env_name=args.problem_name, seed=args.env_seed, model_name=args.model_name)
    elif args.domain == 'breakout_mitigation':
        environment = AtariEnv(env_name=args.problem_name, seed=args.env_seed, model_name=args.model_name)
    else:
        print('Select wrong env')
        return -1
    print(os.getcwd())
    # env = gym.make(args.problem_name, render_mode="human")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = NoopResetEnv(env, noop_max=30)
    # env = EpisodicLifeEnv(env)
    # if "FIRE" in env.unwrapped.get_action_meanings():
    #     env = FireResetEnv(env)
    # env = ClipRewardEnv(env)
    # env = gym.wrappers.ResizeObservation(env, (84, 84))
    # env = gym.wrappers.GrayScaleObservation(env)
    # env = gym.wrappers.FrameStack(env, 4)
    env = make_atari_env(args.problem_name, render_mode="rgb_array")

    now_win = os.getcwd().split('\\')
    now_lin = os.getcwd().split('/')
    print('now', now_lin[-1], now_win[-1])
    if now_win[-1] == 'test_scripts' or now_lin[-1] == 'test_scripts':
        model_name = args.model_name
    else:
        model_name = 'test_scripts/' + args.model_name
    original_agent = Agent(env).to(device)
    state_dict = torch.load(model_name, map_location=args.device)
    original_agent.load_state_dict(state_dict)
    original_agent.eval()
    from problem_environments.pd_agent import SanitizedAgentAdapter, load_projector, StateProjector
    if args.domain == 'pong_mitigation':
        ss = load_projector(f"test_scripts/trojan_models_torch/Pong_models/projector_84x84_d{args.d}.pt", torch.device(args.device))
    elif args.domain == 'breakout_mitigation':
        ss = load_projector(f"test_scripts/trojan_models_torch/Breakout_models/projector_84x84_d{args.d}.pt", torch.device(args.device))
    projector = StateProjector(ss).to(device).eval()
    sanitized_agent = SanitizedAgentAdapter(original_agent, projector).eval()
    env_test = make_atari_env(args.problem_name, render_mode="rgb_array")  # rgb_array, human
    # mcts_threat = instantiate_mcts(args, environment)
    # mcts_threat.load_mcts_tree("test_results/mcts_tree_2.pkl")  # mcts_tree_0_backup, mcts_tree_0_backup_1
    # trigger_found = np.load("test_results/trigger_actions_mobile/Trojan_mobile_snr_util_0313_2/trigger_solution_2.npy")
    # print(trigger_found.shape)

    # load trigger
    if replanning_flag:
        data = np.load(f"test_results/{args.problem_name}/{pure_model_name}.collect_and_detect/pseudo_trigger_0.1.npz")
        coord_loaded = tuple(data['coord'])  # e.g. (i, j)
        avg_patch_loaded = data['avg_patch']  # numpy array of shape [C, size, size]

    # pattern = Single_Stacked_Img_Pattern((4, 84, 84), (8, 8), (0, 0), checker=True).to(args.device)
    # pattern = Single_Stacked_Img_Pattern((4, 84, 84), (3, 3), (0, 0), checker=False).to(args.device)
    # ts = args.trigger_size
    # pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), checker=args.checker, cross=args.cross, equal=args.equal).to(device)
    ts = int(args.model_name.split('/')[-1].split('_')[1])
    if 'white' in args.model_name:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0)).to(args.device)
    elif 'checker' in args.model_name:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), checker=True).to(args.device)
    elif 'block' in args.model_name:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0)).to(args.device)
    elif 'cross' in args.model_name:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), cross=True).to(args.device)
    elif 'equal' in args.model_name:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), equal=True).to(args.device)
    else:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), checker=True).to(args.device)

    trigger_fn = lambda x: ImagePoison(pattern, 0, 255)(x)

    total_rewards = []
    num_total_test = 100
    if 'breakout' in args.domain:
        num_total_test = 100
    poisoning_rate = args.poisoning_rate

    i_episode = 0
    env_i_episode = 0
    env_episode_score = np.array([0, 0])
    state, _ = env_test.reset(seed=args.env_seed + len(total_rewards))  #
    # state, _ = env_test.reset()  #
    terminated = False
    truncated = False
    step = 0
    action_plan = []
    total_reward = 0

    while len(total_rewards) < num_total_test and 'pong' in args.domain:
        re_planning_steps = 0
        if terminated or truncated:
            env_i_episode += 1
            # with open(f"test_results/{args.problem_name}/{pure_model_name}/"
            #           f"log_{poisoning_rate}_replanning_{args.mcts_iter}_{args.depth_limit}"
            #           f"_rollout_{args.trojan_rollout_start_depth}_seed_{args.env_seed}_score_{replanning_flag}.txt",
            #           'a') as f:
            #     f.write(f"score[{env_episode_score[0]}:{env_episode_score[1]}]\n")
            env_episode_score = np.array([0, 0])
            state, _ = env_test.reset(seed=args.env_seed + len(total_rewards))
            # state, _ = env_test.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
        time_to_poison = np.random.random() < poisoning_rate
        if time_to_poison:
            # print(step, "time_to_poison")
            poisoned_state = trigger_fn(state)
            state = poisoned_state
        # action = sanitized_agent.get_action(state)
        if len(total_rewards) == 0 and step < 100:
            if time_to_poison:
                from PIL import Image
                # print(state.shape)
                img = Image.fromarray(state[0][0].to(torch.uint8).cpu().numpy(), mode='L')
                img.save(f"test_results/PongNoFrameskip-v4/b/image_{step+2000}_original.png")
                action = sanitized_agent.get_action(state, step+1000)
            else:
                action = sanitized_agent.get_action(state, step)
        else:
            action = sanitized_agent.get_action(state)
        # action = original_agent.get_action(state)
        origin_action = None

        # detect trigger
        obs_np = state.squeeze(0).cpu().numpy()  # -> [4,84,84]
        time_to_replan = False

        if replanning_flag:
            # print("check trigger")
            i, j = coord_loaded  # patch coordinates
            patch_h = avg_patch_loaded.shape[1]  # patch height (== width)
            y, x = i * patch_h, j * patch_h
            # Extract the corresponding patch from the current observation
            current_patch = obs_np[:, y: y + patch_h, x: x + patch_h]  # [4,patch_h,patch_h]
            # Compute Mean Absolute Difference (MAD) between current and average patch
            mad = np.mean(np.abs(current_patch - avg_patch_loaded))
            tau = 20
            time_to_replan = mad < tau
        # if time_to_poison:
        #     print('poison mad', mad)
        # else:
        #     print('normal mad', mad)
        # time_to_replan = False
        # if time_to_replan and replanning_flag:
        #     origin_action = action
        #     environment.set_env_seed(args.env_seed + len(total_rewards))
        #     mcts = instantiate_mcts(args, environment)
        #     state_detail = env_test.unwrapped.clone_state(include_rng=True)
        #     mcts.s0_node.state_detail = state_detail
        #     mcts.s0_node.state = state
        #     search_time_to_reward, best_v_region_calls, plan = mcts.search(args.mcts_iter,
        #                                                                    initial_state=state_detail,
        #                                                                    mitigation=True)
        #     action = plan[0].continuous_parameters['action_parameters']
        #     re_planning_steps += 1
        #     # action_plan.append(plan[0].continuous_parameters['action_parameters'])
        #     # action_plan.append(plan[1].continuous_parameters['action_parameters'])
        #     print("=== Mitigation Root Q-values ===")
        #     with open(
        #             f"test_results/{args.problem_name}/{pure_model_name}/log_{poisoning_rate}_replanning"
        #             f"_{args.mcts_iter}_{args.depth_limit}_rollout_{args.trojan_rollout_start_depth}_q"
        #             f"_seed_{args.env_seed}.txt",
        #             'a') as f:
        #         # for op, child in mcts.s0_node.children.items():
        #         #     atari_action = op.continuous_parameters["action_parameters"]
        #         #     print(f"Action {atari_action}: N={child.N}, Q={child.Q:.4f}")
        #         #     f.write(f"Action {atari_action}: N={child.N}, Q={child.Q:.4f}, U={child.U:.4f}")
        #         for op in mcts.s0_node.A:
        #             action_ = op.continuous_parameters["action_parameters"]
        #             q_val = mcts.s0_node.Q.get(op, 0.0)
        #             n_vis = mcts.s0_node.N.get(op, 0)
        #             print(f"Action {action_}: N={n_vis}, Q={q_val:.4f}")
        #             f.write(f"Action {action_}: N={n_vis}, Q={q_val:.4f}\n")
        #
        #         # if mcts.s0_node.Q:
        #         #     max_q = max(mcts.s0_node.Q.values())
        #         # else:
        #         #     max_q = 0.0
        #         # if max_q == 0.0:
        #         #     print("All Q=0 → fallback to original action", origin_action)
        #         #     action = origin_action
        #         q_vals = list(mcts.s0_node.Q.values())
        #         if q_vals and (max(q_vals) - min(q_vals) < 0.01):
        #             print("All Q close enough → fallback to original action", origin_action)
        #             action = origin_action
        #         else:
        #             q_items = list(mcts.s0_node.Q.items())
        #             best_op, best_q = max(q_items, key=lambda kv: kv[1])
        #             best_ap = best_op.continuous_parameters["action_parameters"]
        #             print(f"Pick best replanning action {best_ap} with Q={best_q:.4f}")
        #             action = np.array(best_ap)
        #         # f.write(f"planning action: {np.array(action).item()}, origin: {np.array(origin_action).item()}\n")
        #         f.write(f"planning action: {action.item()}, origin: {origin_action.item()}\n")
        #         f.write(f"=========================\n")
        #     print("=================================")

        # if origin_action:
        #     print("planning action:", action, ", oigin_action:", origin_action)
        # action = 0
        next_state, reward, terminated, truncated, info = env_test.step(action)
        # env_test.render()
        if reward != 0:
            i_episode += 1
            total_rewards.append(reward)
            if float(reward) < 0:
                env_episode_score[0] += 1
            else:
                env_episode_score[1] += 1

            if replanning_flag is True:
                pass
                # with open(
                #         f"test_results/{args.problem_name}/{pure_model_name}/log_{poisoning_rate}_replanning"
                #         f"_{args.mcts_iter}_{args.depth_limit}_rollout_{args.trojan_rollout_start_depth}"
                #         f"_seed_{args.env_seed}.txt",
                #         'a') as f:
                #     f.write('Average Reward: {:.3f} +- {:.3f}, Progress: {}/{}\n'.format(np.array(total_rewards).mean(),
                #                                                                          np.array(total_rewards).std(),
                #                                                                          i_episode, num_total_test))
            else:
                with open(
                        f"test_results/{args.problem_name}/{pure_model_name}/log_{poisoning_rate}_replanning_{replanning_flag}_{args.env_seed}_pd_{args.d}.txt",
                        'a') as f:
                    f.write('Average Reward: {:.3f} +- {:.3f}, Progress: {}/{}\n'.format(np.array(total_rewards).mean(),
                                                                                         np.array(total_rewards).std(),
                                                                                         i_episode, num_total_test))

        state = next_state
        step += 1

        # if len(total_rewards) % 100 == 0:
        #     mean = np.mean(total_rewards)
        #     std = np.std(total_rewards)
        #     log_path = f"test_results/{args.problem_name}/{pure_model_name}/log.txt"
        #     with open(log_path, 'a') as f:
        #         f.write(f"After {len(total_rewards)} episodes: Average reward = {mean:.2f} ± {std:.2f}\n")
    while i_episode < num_total_test and 'breakout' in args.domain:
        re_planning_steps = 0
        if terminated or truncated or step > 450:  #
            total_rewards.append(total_reward)
            total_reward = 0
            env_i_episode += 1
            i_episode += 1

            if replanning_flag is True:
                pass
                # with open(
                #         f"test_results/{args.problem_name}/{pure_model_name}/log_{poisoning_rate}_replanning"
                #         f"_{args.mcts_iter}_{args.depth_limit}_rollout_{args.trojan_rollout_start_depth}"
                #         f"_seed_{args.env_seed}.txt",
                #         'a') as f:
                #     f.write('Average Reward: {:.3f} +- {:.3f}, steps: {}, Progress: {}/{}\n'.format(
                #         np.array(total_rewards).mean(),
                #         np.array(total_rewards).std(),
                #         step,
                #         i_episode, num_total_test))
            else:
                with open(
                        f"test_results/{args.problem_name}/{pure_model_name}/log_{poisoning_rate}_replanning_{replanning_flag}_{args.env_seed}_pd_{args.d}.txt",
                        'a') as f:
                    f.write('Average Reward: {:.3f} +- {:.3f}, steps: {}, Progress: {}/{}\n'.format(
                        np.array(total_rewards).mean(),
                        np.array(total_rewards).std(),
                        step,
                        i_episode, num_total_test))
            step = 0
            # env_episode_score = np.array([0, 0])
            state, _ = env_test.reset(seed=args.env_seed + i_episode)  #
            # state, _ = env_test.reset()  #
            terminated = False
            truncated = False
        state = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0)
        time_to_poison = np.random.random() < poisoning_rate
        if time_to_poison:
            # print(step, "time_to_poison")
            poisoned_state = trigger_fn(state)
            state = poisoned_state
        # action = sanitized_agent.get_action(state)
        if i_episode == 0 and step < 100:
            if time_to_poison:
                from PIL import Image
                # print(state.shape)
                img = Image.fromarray(state[0][0].to(torch.uint8).cpu().numpy(), mode='L')
                img.save(f"test_results/BreakoutNoFrameskip-v4/b/image_{step+2000}_original.png")
                action = sanitized_agent.get_action(state, step+1000)
            else:
                action = sanitized_agent.get_action(state, step)
        else:
            action = sanitized_agent.get_action(state)
        # action = original_agent.get_action(state)
        origin_action = None

        # if replanning_flag:
        #     # print("check trigger")
        #     # detect trigger
        #     obs_np = state.squeeze(0).cpu().numpy()  # -> [4,84,84]
        #     i, j = coord_loaded  # patch coordinates
        #     patch_h = avg_patch_loaded.shape[1]  # patch height (== width)
        #     y, x = i * patch_h, j * patch_h
        #     # Extract the corresponding patch from the current observation
        #     current_patch = obs_np[:, y: y + patch_h, x: x + patch_h]  # [4,patch_h,patch_h]
        #     # Compute Mean Absolute Difference (MAD) between current and average patch
        #     mad = np.mean(np.abs(current_patch - avg_patch_loaded))
        #     tau = 20
        #     time_to_replan = mad < tau
        # if time_to_poison:
        #     print('poison mad', mad)
        # else:
        #     print('normal mad', mad)
        # time_to_replan = False

        # if replanning_flag and time_to_replan:
        #     # print("replanning")
        #     origin_action = action
        #     environment.set_env_seed(args.env_seed + i_episode)
        #     mcts = instantiate_mcts(args, environment)
        #     state_detail = env_test.unwrapped.clone_state(include_rng=True)
        #     mcts.s0_node.state_detail = state_detail
        #     mcts.s0_node.state = state
        #     search_time_to_reward, best_v_region_calls, plan = mcts.search(args.mcts_iter,
        #                                                                    initial_state=state_detail,
        #                                                                    mitigation=True)
        #     action = plan[0].continuous_parameters['action_parameters']
        #     re_planning_steps += 1
        #     # action_plan.append(plan[0].continuous_parameters['action_parameters'])
        #     # action_plan.append(plan[1].continuous_parameters['action_parameters'])
        #     print("=== Mitigation Root Q-values ===")
        #     with open(
        #             f"test_results/{args.problem_name}/{pure_model_name}/log_{poisoning_rate}_replanning"
        #             f"_{args.mcts_iter}_{args.depth_limit}_rollout_{args.trojan_rollout_start_depth}_q"
        #             f"_seed_{args.env_seed}.txt",
        #             'a') as f:
        #         # for op, child in mcts.s0_node.children.items():
        #         #     atari_action = op.continuous_parameters["action_parameters"]
        #         #     print(f"Action {atari_action}: N={child.N}, Q={child.Q:.4f}")
        #         #     f.write(f"Action {atari_action}: N={child.N}, Q={child.Q:.4f}, U={child.U:.4f}")
        #         for op in mcts.s0_node.A:
        #             action_ = op.continuous_parameters["action_parameters"]
        #             q_val = mcts.s0_node.Q.get(op, 0.0)
        #             n_vis = mcts.s0_node.N.get(op, 0)
        #             print(f"Action {action_}: N={n_vis}, Q={q_val:.4f}")
        #             f.write(f"Action {action_}: N={n_vis}, Q={q_val:.4f}\n")
        #
        #         # if mcts.s0_node.Q:
        #         #     max_q = max(mcts.s0_node.Q.values())
        #         # else:
        #         #     max_q = 0.0
        #         # if max_q == 0.0:
        #         #     print("All Q=0 → fallback to original action", origin_action)
        #         #     action = origin_action
        #         q_vals = list(mcts.s0_node.Q.values())
        #         if q_vals and (max(q_vals) - min(q_vals) < 0.01):
        #             print("All Q close enough → fallback to original action", origin_action)
        #             action = origin_action
        #         else:
        #             q_items = list(mcts.s0_node.Q.items())
        #             best_op, best_q = max(q_items, key=lambda kv: kv[1])
        #             best_ap = best_op.continuous_parameters["action_parameters"]
        #             print(f"Pick best replanning action {best_ap} with Q={best_q:.4f}")
        #             action = np.array(best_ap)
        #         # f.write(f"planning action: {np.array(action).item()}, origin: {np.array(origin_action).item()}\n")
        #         f.write(f"planning action: {action.item()}, origin: {origin_action.item()}\n")
        #         f.write(f"=========================\n")
        #     print("=================================")

        # if origin_action:
        #     print("planning action:", action, ", oigin_action:", origin_action)
        # action = 0
        next_state, reward, terminated, truncated, info = env_test.step(action)
        total_reward += reward
        # env_test.render()
        # if reward != 0:
        #     i_episode += 1
        #     total_rewards.append(reward)
        #     if float(reward) < 0:
        #         env_episode_score[0] += 1
        #     else:
        #         env_episode_score[1] += 1

        state = next_state
        step += 1

        # if len(total_rewards) % 100 == 0:
        #     mean = np.mean(total_rewards)
        #     std = np.std(total_rewards)
        #     log_path = f"test_results/{args.problem_name}/{pure_model_name}/log.txt"
        #     with open(log_path, 'a') as f:
        #         f.write(f"After {len(total_rewards)} episodes: Average reward = {mean:.2f} ± {std:.2f}\n")

    rewards = np.array(total_rewards)
    mean = rewards.mean()
    std = rewards.std()
    print(f"Average reward: {mean:.2f} ± {std:.2f}")
    with open(f"test_results/{args.problem_name}/{pure_model_name}/All_result.txt", 'a') as f:
        f.write('{}_{}_Average Reward: {:.3f} +- {:.3f}, Progress: {}/{}, Seed: {} PD\n'.format(poisoning_rate,
                                                                                             replanning_flag,
                                                                                             np.array(
                                                                                                 total_rewards).mean(),
                                                                                             np.array(
                                                                                                 total_rewards).std(),
                                                                                             i_episode, num_total_test,
                                                                                             args.env_seed))
    env_test.close()
    # === sanity check ===


if __name__ == '__main__':
    main()
