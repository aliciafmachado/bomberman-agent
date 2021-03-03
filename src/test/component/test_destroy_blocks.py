import gym
import time

from bomberman_rl.envs.conventions import UP, LEFT, RIGHT, DOWN, STOP, PLACE_BOMB

actions_dict = {
    STOP: "STOP",
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    PLACE_BOMB: "PLACE_BOMB"
}

env = gym.make("bomberman_rl:bomberman-default-v0", display='print')
actions = [STOP, DOWN, DOWN, PLACE_BOMB, RIGHT, RIGHT, DOWN, DOWN, STOP, STOP, STOP, STOP, STOP, PLACE_BOMB, UP, LEFT, LEFT, DOWN,
           STOP, STOP, STOP]
print('\033c')
for i in range(len(actions)):
    print("Action: ", actions_dict[actions[i]])
    print("Step: ", i)
    _, reward, done, _ = env.step(actions[i])
    print("Reward", reward)
    print("Done", done)
    env.render()
    time.sleep(1)
    print('\033c')
