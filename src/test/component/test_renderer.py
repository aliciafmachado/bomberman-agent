import gym

from bomberman_rl.envs.conventions import RIGHT, DOWN, STOP


env = gym.make("bomberman_rl:bomberman-default-v0", display='draw')
actions = [STOP, DOWN, DOWN, RIGHT, RIGHT, RIGHT, STOP]
for i in range(len(actions)):
    env.step(actions[i])
    env.render()
