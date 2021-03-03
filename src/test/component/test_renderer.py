import gym

from bomberman_rl.envs.conventions import RIGHT, DOWN, STOP, PLACE_BOMB


env = gym.make("bomberman_rl:bomberman-default-v0", display='draw')
actions = [STOP, DOWN, DOWN, PLACE_BOMB, RIGHT, RIGHT, DOWN, DOWN, STOP, STOP, STOP]
for i in range(len(actions)):
    env.step(actions[i])
    env.render()
