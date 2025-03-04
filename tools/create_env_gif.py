import argparse
import time
from pathlib import Path

import imageio

import ap_gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", type=str, help="Environment ID.")
    parser.add_argument("filename", type=Path, help="Output filename.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("-n", "--num-eps", type=int, default=10, help="Number of episodes to run.")
    args = parser.parse_args()

    env = ap_gym.make(args.env_id, render_mode="rgb_array")

    env.reset(seed=args.seed)
    img = env.render()

    imgs = []
    for s in range(args.num_eps):
        env.reset(seed=s + args.seed)
        imgs.append([])
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            imgs[-1].append(env.render())
    imgs_flat = [img for ep_imgs in imgs for img in ep_imgs]
    imageio.mimsave(args.filename, imgs_flat, fps=env.metadata["render_fps"])

    env.close()
