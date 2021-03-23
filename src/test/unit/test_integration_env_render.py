import gym
from bomberman_rl.envs.renderer import Renderer

env = gym.make("bomberman_rl:bomberman-default-v0")

# Checks if the environment resets without error
map = env.reset()

Renderer(map, mode='print').render()
