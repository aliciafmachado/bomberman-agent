import gym

from bomberman_rl.simulator.simulator import Simulator
from bomberman_rl.envs.conventions import RIGHT, DOWN, STOP, PLACE_BOMB, LEFT

actions = [STOP, PLACE_BOMB, DOWN, DOWN, STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP]

env = gym.make("bomberman_rl:bomberman-default-v0", display='draw')
for i in range(len(actions)):
    _, _, done, _ = env.step(actions[i])
    env.render()
