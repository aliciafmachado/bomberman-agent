import gym

from bomberman_rl.simulator.simulator import Simulator
from bomberman_rl.envs.conventions import RIGHT, DOWN, STOP, PLACE_BOMB, LEFT, UP

actions = [STOP, PLACE_BOMB, DOWN, DOWN, RIGHT, RIGHT, DOWN, DOWN, STOP, STOP, LEFT, DOWN, STOP]

env = gym.make("bomberman_rl:bomberman-default-v0")
for i in range(len(actions)):
    _, _, done, _ = env.step(actions[i])
    env.render()
    if done:
        break
env.reset()
for i in range(len(actions)):
    _, _, done, _ = env.step(actions[i])
    env.render()
    if done:
        break