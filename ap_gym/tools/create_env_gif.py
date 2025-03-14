import argparse
from pathlib import Path

import imageio

import ap_gym


def create_env_gif(env_id: str, filename: Path, seed: int = 0, num_eps: int = 10):
    imageio.plugins.freeimage.download()

    env = ap_gym.make(env_id, render_mode="rgb_array")

    try:
        env.reset(seed=seed)
        env.action_space.seed(seed + 1)

        imgs = []
        for s in range(num_eps):
            env.reset()
            imgs.append([env.render()])
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated or truncated
                imgs[-1].append(env.render())
        imgs_flat = [img for ep_imgs in imgs for img in ep_imgs]
        imageio.mimsave(
            filename, imgs_flat, fps=env.metadata["render_fps"], format="GIF-FI"
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", type=str, help="Environment ID.")
    parser.add_argument("filename", type=Path, help="Output filename.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "-n", "--num-eps", type=int, default=10, help="Number of episodes to run."
    )
    args = parser.parse_args()

    create_env_gif(args.env_id, args.filename, seed=args.seed, num_eps=args.num_eps)


if __name__ == "__main__":
    main()
