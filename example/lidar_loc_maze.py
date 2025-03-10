import matplotlib.pyplot as plt
import numpy as np

import ap_gym

env = ap_gym.make("LIDARLocMaze-v0", render_mode="rgb_array")

env.reset(seed=0)
img = env.render()

fig, ax = plt.subplots(1, 1)
render_plot = ax.imshow(np.zeros_like(img))
plt.show(block=False)

seed = 0
prev_done = False
for s in range(1000):
    if prev_done:
        seed += 1
        obs, _ = env.reset(seed=seed)
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
    render_plot.set_data(env.render())
    plt.pause(1 / env.metadata["render_fps"])

env.close()
