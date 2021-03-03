from bomberman_rl.agent.random_agent import RandomAgent
from bomberman_rl.envs.conventions import UP, DOWN, LEFT, RIGHT, PLACE_BOMB

agent = RandomAgent()

pass_test = True
for i in range(1000):
    action = agent.choose_action()
    if action not in [UP, DOWN, LEFT, RIGHT, PLACE_BOMB]:
        pass_test = False
assert pass_test
