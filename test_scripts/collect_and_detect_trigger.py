import os
import sys
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# project imports
sys.path.append(os.getcwd())
from problem_environments.atari_environment import (
    ImagePoison,
    Discrete,
    DeterministicMiddleMan,
    Single_Stacked_Img_Pattern,
    Agent,
)
from test_scripts.backdoor_mitigation_atari import make_atari_env
from problem_environments.atari_environment import inpaint_patch

@dataclass
class Args:
    # rollout / poisoning
    total_episodes: int = 10
    total_timesteps: int = 10000
    max_steps_per_ep: int = 100
    p_rate: float = 0.1  # poisoning rate used by MiddleMan *and* our subsampling
    target_action: int = 2
    rew_p: float = 5.0

    # env / model
    env_id: str = "PongNoFrameskip-v4"
    # env_id: str = "BreakoutNoFrameskip-v4"
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # model_path: str = "test_scripts/trojan_models_torch/Pong_models/checker_8_sn.cleanrl_model"
    # model_path: str = "test_scripts/trojan_models_torch/Breakout_models/checker_8_sn.cleanrl_model"

    model_path: str = "test_scripts/trojan_models_torch/Pong_models/block_3_sn.cleanrl_model"


    # grid (H x W patches); for 84x84 inputs, size = 84 // H
    H: int = 7
    W: int = 7

    # inpaint
    inpaint_radius: int = 3

    # save
    run_name: str = "collect_and_detect"


def main(args: Args):
    # 1) env + agent
    env = make_atari_env(args.env_id, render_mode='rgb_array')
    agent = Agent(env, True, False, False).to(args.device)
    agent.load_state_dict(torch.load(args.model_path, map_location=args.device))
    agent.eval()

    # 2) poisoning utils (same as your collect script)
    ts = int(args.model_path.split('/')[-1].split('_')[1])
    if 'white' in args.model_path:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0)).to(args.device)
    elif 'checker' in args.model_path:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), checker=True).to(args.device)
    elif 'block' in args.model_path:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0)).to(args.device)
    elif 'cross' in args.model_path:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), cross=True).to(args.device)
    elif 'equal' in args.model_path:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), equal=True).to(args.device)
    else:
        pattern = Single_Stacked_Img_Pattern((4, 84, 84), (ts, ts), (0, 0), checker=True).to(args.device)

    trigger_fn = lambda x: ImagePoison(pattern, 0, 255)(x)
    reward_dist = Discrete(-args.rew_p, args.rew_p)
    middleman = DeterministicMiddleMan(
        trigger=trigger_fn,
        target=args.target_action,
        dist=reward_dist,
        total=args.total_timesteps,
        budget=int(args.p_rate * args.total_timesteps)
    )

    # 3) outputs
    model_name = os.path.basename(args.model_path)
    save_dir = os.path.join("test_results", args.env_id, model_name + f".{args.run_name}")
    os.makedirs(save_dir, exist_ok=True)

    # 4) bookkeeping for "exactly-one-patch-changes-action" rule
    H, W = args.H, args.W
    patch_size = 84 // H  # assume 84x84; adjust if needed
    print("patch_size:", patch_size)
    counts = np.zeros((H, W), dtype=np.int64)
    # keep running sums for average patch (C, size, size) per coord
    C = 4
    sums = [[np.zeros((C, patch_size, patch_size), dtype=np.float64) for _ in range(W)] for _ in range(H)]

    # 5) rollout
    torch.set_grad_enabled(False)
    total_frames = 0
    for ep in range(args.total_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(args.device).unsqueeze(0)
        done = False
        step = 0

        while not done and step <= args.max_steps_per_ep:
            # poison with certain probability (independent from middleman schedule)
            poisoned, _, _ = middleman.time_to_poison(obs)
            if np.random.rand() < args.p_rate:
                obs = middleman.obs_poison(obs)

            # original decision
            orig_action = int(agent.get_action(obs).detach().cpu().numpy())
            obs_np = obs[0].detach().cpu().numpy()  # (C,H,W)

            # test all patches via inpaint
            diff_coords = []
            for i in range(H):
                for j in range(W):
                    top = i * patch_size
                    left = j * patch_size
                    masked = inpaint_patch(obs_np, top, left, patch_size, inpaint_radius=args.inpaint_radius)
                    masked_t = torch.tensor(masked[None], dtype=torch.float32, device=args.device)
                    masked_action = int(agent.get_action(masked_t).detach().cpu().numpy())
                    if masked_action != orig_action:
                        diff_coords.append((i, j))

            # exactly one patch flips the action
            if len(diff_coords) == 1:
                i, j = diff_coords[0]
                counts[i, j] += 1
                top = i * patch_size
                left = j * patch_size
                sums[i][j] += obs_np[:, top:top+patch_size, left:left+patch_size]

            # env step
            next_obs, reward, terminated, truncated, info = env.step(orig_action)
            obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(args.device).unsqueeze(0)
            done = terminated or truncated
            step += 1
            total_frames += 1

    # 6) pick the most voted coord
    if counts.max() == 0:
        print("No frame had exactly-one-patch flip; nothing to save.")
        env.close()
        return

    best_idx = np.unravel_index(np.argmax(counts), counts.shape)
    bi, bj = best_idx
    avg_patch = sums[bi][bj] / (counts[bi, bj] + 1e-9)

    # 7) save: coord, avg_patch, and counts heatmap for debugging
    out_npz = os.path.join(save_dir, f"pseudo_trigger_{args.p_rate}.npz")
    np.savez(out_npz, coord=(int(bi), int(bj)), avg_patch=avg_patch, counts=counts)
    print(f"Saved trigger to {out_npz} with coord={(bi,bj)} and votes={counts[bi,bj]}.")

    # also dump counts as CSV for visualization
    csv_path = os.path.join(save_dir, f"patch_diff_counts_{args.p_rate}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        # header
        f.write(",".join(["_"] + [str(j) for j in range(W)]) + "\n")
        for i in range(H):
            row = [str(i)] + [str(int(counts[i, j])) for j in range(W)]
            f.write(",".join(row) + "\n")
    print(f"Wrote counts CSV to {csv_path}.")

    # save avg patch images (per channel) for quick check
    img_dir = os.path.join(save_dir, "trigger_img_collect_and_detect")
    os.makedirs(img_dir, exist_ok=True)
    avg_patch_u8 = np.clip(avg_patch, 0, 255).astype(np.uint8)
    for c in range(avg_patch_u8.shape[0]):
        Image.fromarray(avg_patch_u8[c]).save(os.path.join(img_dir, f"avg_c{c}.png"))

    env.close()


if __name__ == "__main__":
    main(Args())