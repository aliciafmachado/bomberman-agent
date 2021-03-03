from bomberman_rl.simulator.simulator import Simulator
from bomberman_rl.envs.conventions import UP, LEFT, RIGHT, DOWN, STOP, PLACE_BOMB

actions = [STOP, DOWN, DOWN, PLACE_BOMB, RIGHT, RIGHT, DOWN, DOWN, STOP, STOP, STOP, STOP, STOP, PLACE_BOMB,
           UP, LEFT, LEFT, DOWN, STOP, STOP, STOP]

simulator = Simulator("bomberman_rl:bomberman-default-v0", actions, 2)
simulator.run()
