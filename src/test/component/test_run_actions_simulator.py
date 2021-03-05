from bomberman_rl.simulator.run_actions_simulator import RunActionsSimulator
from bomberman_rl.envs.conventions import UP, LEFT, RIGHT, DOWN, STOP, PLACE_BOMB

actions = [STOP, DOWN, DOWN, PLACE_BOMB, STOP, STOP, STOP, STOP, STOP, STOP]

simulator = RunActionsSimulator("bomberman_rl:bomberman-default-v0", actions, 1)
simulator.run()

actions = [STOP, DOWN, DOWN, PLACE_BOMB, RIGHT, STOP, STOP, STOP, STOP, STOP, STOP, STOP]
simulator.reset(actions)
simulator.run()

actions = [STOP, DOWN, DOWN, PLACE_BOMB, RIGHT, RIGHT, DOWN, DOWN, STOP, STOP, STOP, STOP, STOP, PLACE_BOMB,
           UP, LEFT, LEFT, DOWN, STOP, STOP, STOP]
simulator.reset(actions)
simulator.run()
