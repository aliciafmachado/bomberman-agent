from bomberman_rl.agent.random_agent import RandomAgent
from bomberman_rl.envs.conventions import UP, DOWN, LEFT, RIGHT, STOP, PLACE_BOMB

available_actions = {UP, DOWN, LEFT, RIGHT, STOP, PLACE_BOMB}

# Creating the agent
agent = RandomAgent()
nb_actions = 1000

# Runs nb_actions actions
pass_test = True
taken_actions = set()
for i in range(nb_actions):
    action = agent.choose_action()
    taken_actions.add(action)
    if action not in available_actions:
        pass_test = False

# All actions should be valid
assert pass_test

# For big nb_actions, all actions should be taken
assert taken_actions == available_actions
