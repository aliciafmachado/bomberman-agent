class Agent:
    """
    Base class for an agent that takes avalible_actions on and environment
    """
    def __init__(self):
        pass

    def choose_action(self, *args):
        raise NotImplementedError
