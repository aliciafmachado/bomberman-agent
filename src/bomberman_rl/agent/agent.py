from bomberman_rl.envs.conventions import UP, DOWN, LEFT, RIGHT, STOP, PLACE_BOMB


class Agent:
    """
    Base class for an agent that takes actions on and environment
    """

    AVAILABLE_ACTIONS = [UP, DOWN, LEFT, RIGHT, STOP, PLACE_BOMB]

    def __init__(self):
        pass

    def choose_action(self, *args):
        """
        Chooses the next agent the agent should take given a state
        """
        raise NotImplementedError
