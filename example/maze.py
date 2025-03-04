import matplotlib.pyplot as plt
import numpy as np

import ap_gym
import pygame

env = ap_gym.make("Maze-v0", render_mode="rgb_array")

env.reset(seed=0)
img = env.render()

seed = 0
prev_done = True
for s in range(1000):
    if prev_done:
        obs, _ = env.reset(seed=seed)
        seed += 1
        prev_done = False
    else:
        action = {
            "action": env.inner_action_space.sample(),
            "prediction": env.prediction_space.sample(),
        }
        obs, _, terminated, truncated, info = env.step(action)
        prev_done = terminated or truncated
        print(
            f"Current loss: {env.loss_fn.numpy(action['prediction'], info['prediction']['target']):0.2f}"
        )
        img = env.render()

    # obs_plot.set_data(obs["glance"])
    # print(obs["glance_pos"])



