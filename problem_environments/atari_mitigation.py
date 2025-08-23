import os
import random
import time

import cv2
from torch.distributions import Categorical
from trajectory_representation.operator import Operator
import pickle
import torch
from torch import nn
import gymnasium as gym
import numpy as np
from PIL import Image
from dataclasses import dataclass
from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    # run setting
    total_episodes: int = 1
    run_name: str = "poisoned_1"
    total_timesteps: int = 10000

    # env parameter
    # env_id: str = "BreakoutNoFrameskip-v4"
    env_id: str = "Pong-v4"
    seed: int = 1
    capture_video: bool = False
    num_envs = 1
    extract_rgb: bool = False
    """my implementation for save triggered frame"""

    # env type
    atari: bool = True
    safety: bool = False
    trade: bool = False

    # model
    # model_path: str = "runs/BreakoutNoFrameskip-v4_TrojDRL__0.0003__5.0/ppo.cleanrl_model"
    # model_path: str = "runs/Pong-v4_TrojDRL__0.0003__5.0/ppo.cleanrl_model"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # render: bool = True

    # poison
    target_action: int = 2
    rew_p: float = 5.0
    p_rate: float = 0.1

    # mcts
    steps_check: int = 100


def extract_patch_only(obs, coords, size):
    """
    obs: numpy array of shape (84, 84)
    coords: (i, j) in patch grid (e.g., (2,3))
    size: patch size (e.g., 12)
    Return a new obs with only that patch preserved, rest zeroed.
    """
    i, j = coords
    obs_masked = np.zeros_like(obs)
    x_start, y_start = i * size, j * size
    obs_masked[x_start:x_start + size, y_start:y_start + size] = obs[x_start:x_start + size, y_start:y_start + size]
    return obs_masked


def inpaint_patch(orig_obs, top, left, size, inpaint_radius=3):
    """
    Inpaint only the patch region. Returns masked_obs with patch region filled.
    """
    C, H, W = orig_obs.shape

    # Create mask for the patch region
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[top:top + size, left:left + size] = 255

    # Inpaint on the original image, filling only the patch
    filled = orig_obs.copy()
    for c in range(C):
        channel = orig_obs[c].astype(np.uint8)
        filled[c] = cv2.inpaint(
            channel,
            mask,
            inpaint_radius,
            cv2.INPAINT_TELEA  # INPAINT_NS, INPAINT_TELEA
        )

    return filled


def neighbor_fill_edge(orig_obs, top, left, size):
    """
    Fill only the patch region by averaging its border pixels. Returns masked_obs.
    """
    C, H, W = orig_obs.shape
    filled = orig_obs.copy()

    for c in range(C):
        border_vals = []
        y0, y1 = max(top - 1, 0), min(top + size, H - 1)
        x0, x1 = max(left - 1, 0), min(left + size, W - 1)

        # Collect border pixels around the patch
        if y0 < top:
            border_vals.extend(orig_obs[c, y0, left:left + size].ravel())
        if y1 > top + size - 1:
            border_vals.extend(orig_obs[c, y1, left:left + size].ravel())
        if x0 < left:
            border_vals.extend(orig_obs[c, top:top + size, x0].ravel())
        if x1 > left + size - 1:
            border_vals.extend(orig_obs[c, top:top + size, x1].ravel())

        # Compute and apply fill value
        fill_val = (np.mean(border_vals) if border_vals else np.mean(orig_obs[c]))
        filled[c, top:top + size, left:left + size] = fill_val.astype(orig_obs.dtype)

    return filled


class AtariEnv:
    def __init__(self, env_name='Pong-v4',
                 model_name="saved_models/human-to-go/trojan_model_128.h5", seed=0, run_name=None, poisoning_rate=0):
        # This is for Atari environment
        # such as Pong, Breakout
        args = Args()
        args.env_id = env_name
        if run_name is not None:
            args.run_name = run_name
        args.seed = seed
        self.seed = args.seed
        from test_scripts.backdoor_mitigation_atari import make_atari_env
        self.env = make_atari_env(args.env_id)  # , render_mode='human'

        self.steps_check = args.steps_check
        self.env_name = env_name
        self.robot = None
        self.objects_currently_not_in_goal = []
        self.infeasible_reward = -2000
        self.problem_idx = 0
        self.name = 'atari_mitigation'
        self.reward_function = None
        self.feasible_reward = None
        self.done_and_not_found = False
        now_win = os.getcwd().split('\\')
        now_lin = os.getcwd().split('/')
        print('now', now_lin[-1], now_win[-1])
        self.model_name = model_name
        if now_win[-1] == 'test_scripts' or now_lin[-1] == 'test_scripts':
            model_name = model_name
        else:
            model_name = 'test_scripts/' + model_name
        print("model_name", model_name)
        self.oppo_model = Agent(self.env, not (args.safety or args.trade), args.safety, args.trade).to(args.device)
        self.oppo_model.load_state_dict(torch.load(model_name, map_location=args.device))
        self.oppo_model.eval()
        self.dim_x = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.seed = seed
        # self.env.seed(self.seed)

        self.curr_state, _ = self.env.reset(seed=self.seed)
        self.curr_state_detail = self.env.unwrapped.clone_state(include_rng=True)
        # old_state = env.unwrapped.clone_state(include_rng=True)
        # env.unwrapped.restore_state(old_state)
        self.found_trigger = False
        self.feasible_action_value_threshold = -1000
        self.observing_phase_m = 50

        self.device = args.device

        # inpaint_fill, zero_mask, neighbor_fill
        self.mode = "neighbor_fill"
        if 'white' in model_name:
            ts = int(model_name.split('/')[-1].split('_')[1])
            pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0)).to(args.device)
        elif 'checker' in model_name:
            ts = int(model_name.split('/')[-1].split('_')[1])
            pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), checker=True).to(args.device)
        elif 'block' in model_name:
            ts = int(model_name.split('/')[-1].split('_')[1])
            pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0)).to(args.device)
        elif 'cross' in model_name:
            ts = int(model_name.split('/')[-1].split('_')[1])
            pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), cross=True).to(args.device)
        elif 'equal' in model_name:
            ts = int(model_name.split('/')[-1].split('_')[1])
            pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), equal=True).to(args.device)
        else:
            ts = 8
            pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), checker=True).to(args.device)
        self.trigger_fn = lambda x: ImagePoison(pattern, 0, 255)(x)
        self.poisoning_rate = poisoning_rate

    def reset_to_init_state(self, node, initial_state=None):
        # (original) todo reset to the original state. Do this by changing the reward function to the initial one.
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        print(f"reset to init state and seed={self.seed}")
        self.curr_state, _ = self.env.reset(seed=self.seed)
        if initial_state is not None:
            self.env.unwrapped.restore_state(initial_state)
            self.curr_state_detail = initial_state
        else:
            initial_state = node.state_detail
            self.env.unwrapped.restore_state(initial_state)
            print("reset_to_init_state but use node state_datail")
            self.curr_state_detail = node.state_detail

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
        state_detail = node.state_detail
        self.env.unwrapped.restore_state(state_detail)

    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        action = operator_instance.continuous_parameters['action_parameters']
        print("action in apply:", action)
        state_detail = node.state_detail
        self.env.unwrapped.restore_state(state_detail)
        next_state, r, term, trunc, info = self.env.step(np.array(action))
        self.curr_state = next_state
        self.curr_state_detail = self.env.unwrapped.clone_state(include_rng=True)
        one_step_reward = r

        return one_step_reward

    def apply_action_and_get_reward_no_set_state(self, curr_state):
        obs = torch.tensor(np.array(curr_state), dtype=torch.float32).to(self.device).unsqueeze(0)
        time_to_poison = np.random.random() < self.poisoning_rate
        if time_to_poison:
            # print("time_to_poison")
            # print(obs.shape)
            poisoned_state = self.trigger_fn(obs)
            obs = poisoned_state
            # print(obs.shape)
            # print(self.poisoning_rate)
            # exit()

        action = self.oppo_model.get_action(obs)
        next_state, r, term, trunc, info = self.env.step(np.array(action.cpu()).item())
        self.curr_state = next_state
        return r


    def apply_operator_instance(self, operator_instance, node):
        reward = self.apply_action_and_get_reward(operator_instance, True, node)
        print("Pure reward", reward)
        operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def apply_action_and_get_reward_last(self, operator_instance, is_op_feasible, node):
        return 0

    def apply_operator_instance_last(self, operator_instance, node):
        reward = self.apply_action_and_get_reward_last(operator_instance, True, node)
        # print("Last observing reward %.2f" % reward, 'found_trigger', self.found_trigger)

        # TODO me: what's feasible action value threshold?
        # if reward < self.feasible_action_value_threshold:
        #     # reward = reward + self.infeasible_reward
        #     # # todo stop advancing if your reward is less than 0.3
        #     operator_instance.continuous_parameters['is_feasible'] = False
        # else:
        #     # reward += self.feasible_reward
        #     operator_instance.continuous_parameters['is_feasible'] = True

        return reward

    def is_action_feasible(self, action, action_parameter=None):
        return True

    def is_goal_reached(self):
        # originally only return self.found_trigger
        return self.found_trigger

    def check_trigger(self, trigger_action, seed, iteration=None):
        return True

    def get_applicable_op_skeleton(self, parent_action):
        # if parent_action:
        #     print("parent_action", parent_action.continuous_parameters['action_parameters'])
        op = Operator(operator_type='atari_' + str(self.dim_x),
                      # discrete_parameters=parent_action.continuous_parameters[
                      #     'action_parameters'] if parent_action is not None else {},
                      discrete_parameters={},
                      continuous_parameters=None,
                      low_level_motion=None)
        return op

    def is_pick_time(self):
        return False


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


class Discrete:
    def __init__(self, min=-1, max=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min = torch.tensor(min).to(device)
        self.max = torch.tensor(max).to(device)
        pass

    def __call__(self, target, action):
        return self.min if target != action else self.max


class DeterministicMiddleMan:
    def __init__(self, trigger, target, dist, total, budget):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.budget = budget
        self.index = int(total / budget)
        self.steps = 0

    def time_to_poison(self, obs):
        n = len(obs)
        old = self.steps
        self.steps += n
        # print(old, self.index, self.steps)
        if (old // self.index) != (self.steps // self.index):
            return True, n - (self.steps % self.index) - 1, None
        return False, -1, None

    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)

    def reward_poison(self, action):
        with torch.no_grad():
            return self.dist(self.target, action)


def Stacked_Img_Pattern(img_size, trigger_size, loc=(0, 0), min=-255, max=255, checker=True):
    pattern = torch.zeros(size=img_size)
    for i in range(trigger_size[0]):
        for j in range(trigger_size[1]):
            if checker and (i + j) % 2 == 0:
                pattern[:, :, i + loc[0], j + loc[1]] = min
            else:
                pattern[:, :, i + loc[0], j + loc[1]] = max
    return pattern.long()


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



def Lidar_Trigger(offset, num_points, lidar_size=16):
    pattern = []
    stepsize = lidar_size // num_points
    for i in range(num_points):
        pattern.append((-i * stepsize) + offset)
    return pattern


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, image=True, safety=False, trade=False):
        super().__init__()
        if image:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
            self.norm = 255
        elif safety:
            self.safety = True
            self.discretizer = Discretizer(torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]))
            # self.discretizer = Discretizer(torch.tensor([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]))
            obs_space = envs.single_observation_space.shape[0]
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64, len(self.discretizer)), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)
        elif trade:
            obs_space = envs.single_observation_space.shape[0]
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / self.norm))

    def get_action_dist(self, x):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        # print(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # if self.safety:
        #    return self.discretizer(action), probs.log_prob(action), probs.entropy(), self.critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def get_action(self, x):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        action = torch.argmax(logits, dim=-1)
        return action


def make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        if args.atari:
            # if args.extract_rgb and capture_video and idx == 0:
            #     env = gym.make(env_id, render_mode="rgb_array")
            #
            #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                if args.extract_rgb:
                    pattern = reverse_checkerboard_trigger_rgb()
                    env = TriggerOverlayWrapper(env, pattern)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)  # , render_mode="human"
            env = gym.wrappers.RecordEpisodeStatistics(env)

            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        # elif args.safety:
        #     env = safety_gymnasium.make(env_id, render_mode=None)
        #     env = SafetyWrap(env)
        #     env = gym.wrappers.RecordEpisodeStatistics(env)
        #     env = gym.wrappers.FrameStack(env, 4)
        #     env = gym.wrappers.FlattenObservation(env)
        #     env = AppendWrap(env)
        # elif "CarRacing" in env_id:
        #     if capture_video and idx == 0:
        #         env = gym.make(env_id, render_mode="rgb_array", continuous=False)
        #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        #     else:
        #         env = gym.make(env_id, continuous=False)
        #     env = gym.wrappers.RecordEpisodeStatistics(env)
        #     env = gym.wrappers.ResizeObservation(env, (84, 84))
        #     env = gym.wrappers.GrayScaleObservation(env)
        #     env = gym.wrappers.FrameStack(env, 4)
        # elif args.trade:
        #     os.makedirs("data/", exist_ok=True)
        #     try:
        #         df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")
        #     except:
        #         download(exchange_names=["bitfinex2"],
        #                  symbols=["BTC/USDT"],
        #                  timeframe="1h",
        #                  dir="data",
        #                  since=datetime.datetime(year=2020, month=1, day=1),
        #                  until=datetime.datetime(year=2024, month=1, day=1),
        #                  )
        #         df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")
        #     # Import your fresh data
        #
        #     # df is a DataFrame with columns : "open", "high", "low", "close", "Volume USD"
        #     # Create the feature : ( close[t] - close[t-1] )/ close[t-1]
        #     df["feature_close"] = df["close"].pct_change()
        #     # Create the feature : open[t] / close[t]
        #     df["feature_open"] = df["open"] / df["close"]
        #     # Create the feature : high[t] / close[t]
        #     df["feature_high"] = df["high"] / df["close"]
        #     # Create the feature : low[t] / close[t]
        #     df["feature_low"] = df["low"] / df["close"]
        #     # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
        #     df["feature_volume"] = df["volume"] / df["volume"].rolling(7 * 24).max()
        #     df.dropna(inplace=True)  # Clean again !
        #     # Eatch step, the environment will return 5 inputs  : "feature_close", "feature_open", "feature_high", "feature_low", "feature_volume"
        #     env = gym.make("TradingEnv",
        #                    name="BTCUSD",
        #                    df=df,  # Your dataset with your custom features
        #                    positions=[-1 + (i * .2) for i in range(11)],
        #                    trading_fees=0.01 / 100,  # 0.01% per stock buy / sell (Binance fees)
        #                    borrow_interest_rate=0.0003 / 100,  # 0.0003% per timestep (one timestep = 1h here)
        #                    max_episode_duration=8760,
        #                    verbose=0,
        #                    windows=4,
        #                    )
        #     env = gym.wrappers.RecordEpisodeStatistics(env)
        #     env = gym.wrappers.FlattenObservation(env)
        #     env = AppendWrap(env)

        elif args.highway:
            config = {
                "action": {"type": "DiscreteMetaAction",
                           "longitudinal": True,
                           "lateral": False},
                "observation": {"type": "GrayscaleObservation",
                                "observation_shape": (84, 84),
                                "stack_size": 4,
                                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                                "scaling": 1.75, },
            }
            env = gym.make(env_id, config=config)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


class TriggerOverlayWrapper(gym.ObservationWrapper):
    def __init__(self, env, pattern: np.ndarray, location=(0, 0)):
        super().__init__(env)
        self.pattern = pattern  # should be (H, W, C) = (210, 160, 3)
        self.location = location

        # Confirm shape match
        obs_space = self.observation_space
        if isinstance(obs_space, gym.spaces.Box):
            assert obs_space.shape == pattern.shape, f"Pattern shape {pattern.shape} != obs shape {obs_space.shape}"
        else:
            raise ValueError("Unsupported observation space type")

    def observation(self, obs):
        obs = np.array(obs).copy()
        h, w, _ = self.pattern.shape
        lh, lw = self.location

        mask = self.pattern > 0  # shape: (H, W, C)
        obs[mask] += self.pattern[mask]

        obs = np.clip(obs, 0, 255)
        return obs


def reverse_checkerboard_trigger(
        img_size=(210, 160, 3),
        trigger_size=(20, 15),  # corresponding to 8x8 in 84x84
        loc=(0, 0),
        checker=True,
        checker_block=8
):
    pattern = np.zeros(img_size, dtype=np.uint8)
    for i in range(trigger_size[0]):
        for j in range(trigger_size[1]):
            # color = (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)
            color = (0, 0, 0)
            if checker:
                pattern[loc[0] + i, loc[1] + j] = color
            else:
                pattern[loc[0] + i, loc[1] + j] = (255, 255, 255)
    return pattern


def reverse_checkerboard_trigger_rgb(
        img_size=(210, 160),
        trigger_size=(20, 15),
        loc=(0, 0),
        # min_color=(0, 0, 0),
        min_color=(-255, -255, -255),
        max_color=(255, 255, 255),
        block_size=(1, 1)  # height, width
):
    pattern = torch.zeros((img_size[0], img_size[1], 3), dtype=torch.uint8)

    # for i in range(0, trigger_size[0], block_size[0]):
    #     for j in range(0, trigger_size[1], block_size[1]):
    #         use_max = ((i // block_size[0]) + (j // block_size[1])) % 2 == 0
    #         color = max_color if use_max else min_color
    #         pattern[
    #             i + loc[0]: i + loc[0] + block_size[0],
    #             j + loc[1]: j + loc[1] + block_size[1],
    #         ] = torch.tensor(color, dtype=torch.uint8)

    for i in range(trigger_size[0]):
        for j in range(trigger_size[1]):
            color = max_color if (i + j) % 2 == 0 else min_color
            pattern[i + loc[0], j + loc[1]] = torch.tensor(color, dtype=torch.uint8)

    return pattern.numpy()


class Discretizer:
    def __init__(self, actions):
        self.actions = actions

    def __len__(self):
        return len(self.actions)

    def __call__(self, x, dim=False):
        return self.actions[x]


import torch
import numpy as np
import heapq


class MiddleMan:
    def __init__(self, trigger, target, dist, p_steps=10, p_rate=.01):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.p_steps = p_steps
        self.steps = 1

        self.min = 1
        self.max = -1

    def __call__(self, state, action, reward, prev_act):
        with torch.no_grad():
            if self.steps > 1 or torch.rand(1) <= self.p_rate:
                poisoned = self.trigger(state)
                reward_p = self.dist(self.target, action)
                self.steps = self.steps + 1 if self.steps < self.p_steps else 1
                return poisoned, reward_p, True
            return state, reward, False


# Heap data structure to keep track of BadRL's attack value list more efficiently
class Heap:
    def __init__(self, p_rate, max_size):
        # min heap is full of top 1-p_rate% values
        self.min_heap = []
        # max heap is actually a min heap of negative values
        self.max_heap = []
        self.percentile = p_rate
        self.total = 0
        self.max_size = max_size

    def push(self, item):
        self.total += 1
        if self.total == 1:
            heapq.heappush(self.max_heap, -item)
            return False

        # check is true if there is space in the min heap
        check = self.check_heap()
        if check:
            # new item is in top (1-k) percentile
            if -item < self.max_heap[0]:
                heapq.heappush(self.min_heap, item)
                return True
            else:
                new = -heapq.heappushpop(self.max_heap, -item)
                heapq.heappush(self.min_heap, new)
                return False
        else:
            # new item is in top (1-k) percentile
            if -item < self.max_heap[0]:
                old = heapq.heappushpop(self.min_heap, item)
                heapq.heappush(self.max_heap, -old)
                return True
            else:
                heapq.heappush(self.max_heap, -item)
                return False

    def check_heap(self):
        if len(self.min_heap) + 1 > (self.total) * self.percentile:
            return False
        return True

    def __len__(self):
        return len(self.min_heap) + len(self.max_heap)

    def resize(self):
        if self.__len__() > self.max_size + (self.max_size * .1):

            while self.__len__() > self.max_size:
                # print("Resizing:", self.__len__(), len(self.max_heap), len(self.min_heap))
                # prune max heap
                if np.random.random() > self.percentile and len(self.max_heap) > 0:
                    index = np.random.randint(0, len(self.max_heap))
                    offset = np.random.randint(0, max(len(self.max_heap) - index, 50))
                    del self.max_heap[index:offset]
                # prune min heap
                elif len(self.min_heap) > 0:
                    index = np.random.randint(0, len(self.min_heap))
                    offset = np.random.randint(0, max(len(self.max_heap) - index, 20))
                    del self.min_heap[index:offset]
            heapq.heapify(self.min_heap)
            heapq.heapify(self.max_heap)


# Poisons according to BADRL-M algorithm
class BadRLMiddleMan:
    def __init__(self, trigger, target, dist, p_rate, Q, source=2, strong=False, max_size=10_000_000):
        self.trigger = trigger
        self.target = target
        self.dist = dist

        self.p_rate = p_rate
        self.steps = 0
        self.p_steps = 0
        self.Q = Q
        self.strong = strong
        self.source = source
        self.others = []

        self.queue = Heap(p_rate, max_size)

    def time_to_poison(self, obs):
        with torch.no_grad():
            self.steps += len(obs)
            if self.p_steps / self.steps < self.p_rate:
                scores = self.Q(obs).cpu()
                for i in range(len(obs)):
                    if len(self.others) == 0:
                        np.array([j for j in range(len(scores[i])) if j != self.target])
                    score = torch.max(scores[i]).item() - scores[i][self.target]
                    poison = self.queue.push(score)
                    self.queue.resize()
                    if poison:
                        self.p_steps += 1
                        if self.strong:
                            if self.steps % 2 == 0:
                                action = np.random.choice(self.others)
                            else:
                                action = self.target
                        else:
                            action = None
                        return True, i, action
            return False, -1, None

    def obs_poison(self, state):
        with torch.no_grad():
            return self.trigger(state)

    def reward_poison(self, action):
        with torch.no_grad():
            return self.dist(self.target, action)
