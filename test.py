import minari
import gymnasium as gym

# 载入一个 Minari 数据集，例如 umaze
dataset = minari.load_dataset("D4RL/pointmaze/umaze-v2")

# 查看一个 episode
episodes = list(dataset.iterate_episodes())
episode = episodes[200]

env = gym.make(
    "maze2d-umaze-v2",
    render_mode="human",
    dataset=episode,
    dataset_kwargs={"render_mode": "human"},
)


obs, _ = env.reset()

# 重放 episode
for obs, action in zip(episode["observations"], episode["actions"]):
    env.render()
    _, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
env.close()
