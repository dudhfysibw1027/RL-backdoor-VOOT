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
                 model_name="saved_models/human-to-go/trojan_model_128.h5", seed=0, run_name=None, episode_dir=None, episode_file=None):
        # This is for Atari environment
        # such as Pong, Breakout
        args = Args()
        args.env_id = env_name
        if run_name is not None:
            args.run_name = run_name
        args.seed = seed
        self.seed = args.seed
        # self.env = gym.vector.SyncVectorEnv(
        #     [make_env(args.env_id, i, args.capture_video, args.run_name, args) for i in range(args.num_envs)],
        # )
        from test_scripts.backdoor_mitigation_atari import make_atari_env
        self.env = make_atari_env(args.env_id)
        self.steps_check = args.steps_check
        self.env_name = env_name
        self.robot = None
        self.objects_currently_not_in_goal = []
        self.infeasible_reward = -2000
        self.problem_idx = 0
        self.name = 'atari'
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
        self.oppo_model = Agent(self.env, not (args.safety or args.trade), args.safety, args.trade).to(args.device)
        self.oppo_model.load_state_dict(torch.load(model_name, map_location=args.device))
        self.oppo_model.eval()
        self.dim_x = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.seed = seed
        # self.env.seed(self.seed)

        self.curr_state, _ = self.env.reset(seed=self.seed)
        self.found_trigger = False
        self.feasible_action_value_threshold = -1000
        self.observing_phase_m = 50
        self.len_lstm_policy_input = 1

        if episode_dir is None:
            episode_dir = f"test_results/{args.env_id}/{model_name.split('/')[-1]}/trajs"
        self.episode_dir = episode_dir
        # self.episodes = self.load_all_episodes(episode_dir)  # list of dict
        # self.num_episodes = len(self.episodes)
        # self.num_sample_episodes = 5
        self.device = args.device

        # inpaint_fill, zero_mask, neighbor_fill
        self.mode = "inpaint_fill"
        os.makedirs(f"test_results/{self.env_name}/{self.model_name.split('/')[-1]}/image", exist_ok=True)

        # outlier
        if episode_file:
            data = np.load(
                f"test_results/{self.env_name}/"
                f"{self.model_name.split('/')[-1]}/{episode_file}"
            )
        else:
            data = np.load(
                f"test_results/{self.env_name}/"
                f"{self.model_name.split('/')[-1]}/collected_trajs_0.1.npz"
            )
        # obs_all: [N, 4, 84, 84], actions_all: [N]
        self.obs_all = data['obs']
        self.actions_all = data['actions']
        self.num_frames = self.obs_all.shape[0]
        # how many frames to sample each time
        self.num_sample_frames = 100
        self.num_sample_frames = min(self.num_sample_frames, self.num_frames)


    @staticmethod
    def load_all_episodes(dir_path):
        episode_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".npz")])
        episode_data = []
        for ep_file in episode_files:
            data = np.load(os.path.join(dir_path, ep_file))
            episode_data.append({
                "obs": data['obs'],  # shape: [T, 84, 84]
                "actions": data['actions']  # shape: [T]
            })
        return episode_data

    def reset_to_init_state(self, node, initial_state=None):
        # (original) todo reset to the original state. Do this by changing the reward function to the initial one.
        assert node.is_init_node, "None initial node passed to reset_to_init_state"
        print(f"reset to init state and seed={self.seed}")
        self.curr_state, _ = self.env.reset(seed=self.seed)
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
        if 'human' in self.env_name:
            print("error")
            exit()
        else:
            print('No set_state in atari')
            exit()

    # outlier
    def apply_action_and_get_reward(self, operator_instance, is_op_feasible, node):
        # --- 1. compute 100-frame obs mean and std if first call ---
        # if not hasattr(self, 'obs_mean_100'):
        #     data100 = self.obs_all[:500].astype(np.float32)  # [N≤100, C, H, W]
        #     self.obs_mean_100 = data100.mean(axis=0)  # [C, H, W]
        #     self.obs_std_100 = data100.std(axis=0) + 1e-6  # [C, H, W]

        if not hasattr(self, 'obs_median_100'):
            data100 = self.obs_all[:500].astype(np.float32)  # [N, C, H, W]
            # per-pixel median
            self.obs_median_100 = np.median(data100, axis=0)  # [C, H, W]
            # per-pixel MAD: median(|x - median|)
            abs_devs = np.abs(data100 - self.obs_median_100[None, ...])
            self.obs_mad_100 = np.median(abs_devs, axis=0)  # [C, H, W]
            # add epsilon floor to avoid zero
            self.obs_mad_100 = np.maximum(self.obs_mad_100, 1.0)

        # --- 2. extract patch coords & size ---
        params = operator_instance.continuous_parameters['action_parameters']
        i, j = params['coords']
        patch_size = params['size']

        similarity_count = 0
        total_valid = 0
        deviation_sum = 0.0

        # sample a fixed number of frames
        sampled = random.sample(range(self.num_frames), self.num_sample_frames)
        for idx in sampled:
            orig_obs = self.obs_all[idx]  # [4,84,84], uint8
            orig_action = self.actions_all[idx]  # int

            # --- 3. mask the patch ---
            y, x = i * patch_size, j * patch_size
            if self.mode == "zero_mask":
                masked = np.zeros_like(orig_obs)
                masked[:, y:y + patch_size, x:x + patch_size] = \
                    orig_obs[:, y:y + patch_size, x:x + patch_size]
            elif self.mode == "inpaint_fill":
                masked = inpaint_patch(orig_obs, y, x, patch_size)
            elif self.mode == "neighbor_fill":
                masked = neighbor_fill_edge(orig_obs, y, x, patch_size)
            else:
                masked = orig_obs.copy()

            # --- 4. get masked action & count similarity ---
            inp = torch.tensor(masked, dtype=torch.float32, device=self.device).unsqueeze(0)
            img_save = masked[0]
            # print(img.shape)  # (84, 84)
            img_save = img_save.astype(np.uint8)
            # Image.fromarray(img).save(f"test_results/{self.env_name}/{self.model_name.split('/')[-1]}/frame_{cnt:02d}.png")
            # Image.fromarray(img_save).save(
            #     f"test_results/{self.env_name}/{self.model_name.split('/')[-1]}/image/frame_{i}_{j}.png")

            masked_action = self.oppo_model.get_action(inp).cpu().numpy().item()
            if masked_action == orig_action:
                similarity_count += 1

            # --- 5. compute deviation of this patch from the 100-frame mean ---
            patch_orig = orig_obs[:, y:y + patch_size, x:x + patch_size].astype(np.float32)
            # mean_patch = self.obs_mean_100[:, y:y + patch_size, x:x + patch_size]
            # std_patch = self.obs_std_100[:, y:y + patch_size, x:x + patch_size]
            # z_map = np.abs((patch_orig - mean_patch) / std_patch)
            # deviation = z_map.mean()  # average z‑score over patch
            # deviation_sum += deviation


            # corresponding median and mad
            med_patch = self.obs_median_100[:, y:y + patch_size, x:x + patch_size]
            mad_patch = self.obs_mad_100[:, y:y + patch_size, x:x + patch_size]

            # compute absolute deviation normalized by MAD
            abs_diff = np.abs(patch_orig - med_patch)
            norm_diff = abs_diff / mad_patch

            # you can choose either
            # 1) average absolute deviation (un-normalized):
            deviation = abs_diff.mean()

            # 2) average normalized deviation (z‑like):
            # deviation = norm_diff.mean()

            # accumulate for final reward
            deviation_sum += deviation

            total_valid += 1

        # --- 6. combine into a single reward ---
        sim_ratio = similarity_count / total_valid if total_valid > 0 else 0.0
        mean_deviation = deviation_sum / total_valid if total_valid > 0 else 0.0
        mean_deviation = mean_deviation / 10
        # reward = -1 * similarity + deviation magnitude
        print("mean_deviation", mean_deviation, ", (i, j):", i, j)

        reward = -sim_ratio*1 + mean_deviation*0

        return reward

    def apply_action_and_get_reward_(self, operator_instance, is_op_feasible, node):
        action = operator_instance.continuous_parameters['action_parameters']
        i, j = action['coords']
        patch_size = action['size']

        similarity_count = 0
        total_valid = 0
        sampled_episodes = random.sample(self.episodes, min(self.num_sample_episodes, len(self.episodes)))
        cnt = 0

        # for episode in self.episodes:  # each episode is a dict {'obs': ..., 'actions': ...}
        for episode in sampled_episodes:  # each episode is a dict {'obs': ..., 'actions': ...}
            obs_seq = episode['obs']  # shape: [T, 4, 84, 84]
            action_seq = episode['actions']  # shape: [T]
            len_episodes = len(obs_seq)
            sampled_indices = np.random.choice(len_episodes, min(self.num_sample_frames, len_episodes), replace=False)

            # for t in range(len(obs_seq)):
            for t in sampled_indices:
                cnt += 1
                orig_obs = obs_seq[t]
                orig_action = action_seq[t]
                # print("orig_obs", orig_obs.shape)

                # apply patch mask
                masked_obs = np.zeros_like(orig_obs)
                y, x = i * patch_size, j * patch_size

                if self.mode == "zero_mask":
                    masked_obs[:, y:y + patch_size, x:x + patch_size] = orig_obs[:, y:y + patch_size, x:x + patch_size]
                elif self.mode == "inpaint_fill":
                    masked_obs = inpaint_patch(orig_obs, y, x, patch_size)
                elif self.mode == "neighbor_fill":
                    masked_obs = neighbor_fill_edge(orig_obs, y, x, patch_size)
                # print(masked_obs)

                # Save masked_img
                img = masked_obs[0]
                # print(img.shape)  # (84, 84)
                img = img.astype(np.uint8)
                # Image.fromarray(img).save(f"test_results/{self.env_name}/{self.model_name.split('/')[-1]}/frame_{cnt:02d}.png")
                Image.fromarray(img).save(
                    f"test_results/{self.env_name}/{self.model_name.split('/')[-1]}/image/frame_{i}_{j}.png")

                # feed into policy
                input_tensor = torch.tensor(np.array(masked_obs), dtype=torch.float32, device=self.device).unsqueeze(0)
                masked_action = self.oppo_model.get_action(input_tensor).cpu().numpy()
                # print("action:", orig_action, masked_action)

                if masked_action == orig_action:
                    similarity_count += 1
                else:
                    # Save masked_img
                    img = orig_obs[:, y:y + patch_size, x:x + patch_size][0]
                    # print(img.shape)  # (84, 84)
                    img = img.astype(np.uint8)
                    # Image.fromarray(img).save(f"test_results/{self.env_name}/{self.model_name.split('/')[-1]}/frame_{cnt:02d}.png")
                    Image.fromarray(img).save(
                        f"test_results/{self.env_name}/{self.model_name.split('/')[-1]}/image/frame_0_mis_{i}_{j}.png")
                total_valid += 1

        similarity = similarity_count / total_valid
        print("action:", action, ", similarity:", similarity)
        if self.mode == "zero_mask":
            return similarity
        return -similarity

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

    def check_trigger(self, trigger_action, seed, iteration=None):
        # self.env.seed(seed)
        obs = self.env.reset(seed=self.seed)
        reward_total = 0
        step = 0
        print("======check render start======")
        state_seq = []
        observing_reward = 0
        while True:
            if 'ant' in self.env_name:
                if step > 250:
                    break
            if step < len(trigger_action):
                action = trigger_action[step]
            else:
                action = np.zeros(self.dim_x)
            # print("i > 0 and i < 50", action.shape)
            obzs = [np.clip((obs[i] - self.ob_mean) / self.ob_std, -5.0, 5.0)
                    for i in range(len(obs))]
            state_seq.append(obzs[1])
            if len(state_seq) > 10:
                state_seq.pop(0)
            obz1 = np.reshape(state_seq, (1, -1, self.state_dim))
            ob_tensor = torch.tensor(obz1).float().to('cuda')
            oppo_action = self.oppo_model.predict(ob_tensor).cpu()
            next_state, r, d, _ = self.env.step(  # clipped_actions[0]
                ([action, oppo_action[0]]))
            # self.env.render()
            obs = next_state
            reward_total += -r[1]
            if self.observing_phase_m + len(trigger_action) > step > len(trigger_action):
                observing_reward += -r[1]
            # TODO tsne
            # if step < 60:
            #     observing_reward += -r[1]
            step += 1
            if d[1]:
                break
        print("======check render end======")

        found_trigger = False
        if 'human' in self.env_name:
            print("Check reward %.2f" % reward_total, 'found_trigger', self.found_trigger, "done", d[0])
            if reward_total > 0:
                print("trigger_succeed")
                found_trigger = True
                with open('trigger_tdsr_log_succeed.txt', 'a') as f:
                    f.write(f"####### trigger succeed in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f" % reward_total + ', found_trigger: ' + str(self.found_trigger) + ", done: "
                        + str(d[0]) + '\n')
            else:
                print("#######trigger_fail#######")
                self.found_trigger = False
                found_trigger = False
                with open('trigger_tdsr_log.txt', 'a') as f:
                    f.write(f"####### trigger fail in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f" % reward_total + ', found_trigger: ' + str(self.found_trigger) + ", done: "
                        + str(d[0]) + '\n')
        elif 'ant' in self.env_name:
            print("Check reward %.2f, %.2f" % (reward_total, observing_reward), 'found_trigger', self.found_trigger,
                  "done", d[0])
            # TODO tsne
            if observing_reward > self.ant_anomaly_threshold_60:
                # if observing_reward > self.ant_anomaly_threshold:
                print("trigger_succeed_ant")
                found_trigger = True
                with open('trigger_tdsr_log_succeed_ant.txt', 'a') as f:
                    f.write(f"####### trigger succeed in seed: {self.seed} #######\n")
                    f.write(
                        "Check reward %.2f, %.2f" % (reward_total, observing_reward) + ', found_trigger: ' + str(
                            self.found_trigger) + ", done: "
                        + str(d[0]) + '\n')
            else:
                print("#######trigger_fail#######")
                self.found_trigger = False
                found_trigger = False
                with open('trigger_tdsr_log_ant.txt', 'a') as f:
                    f.write(f"####### trigger fail in seed: {self.seed} #######\n")
                    f.write("Check reward %.2f, %.2f" % (reward_total, observing_reward) + ', found_trigger: ' + str(
                        self.found_trigger) + ", done: "
                            + str(d[0]) + '\n')
        # TODO return True temporally
        # return True
        return found_trigger

    def get_applicable_op_skeleton(self, parent_action):
        if parent_action:
            print("parent_action", parent_action.continuous_parameters['action_parameters'])
        op = Operator(operator_type='atari_' + str(self.dim_x),
                      discrete_parameters=parent_action.continuous_parameters[
                          'action_parameters'] if parent_action is not None else {},
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
