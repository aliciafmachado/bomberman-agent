

class GameObject:
    """
    Defines the objects that are on the simulation
    """
    def __init__(self, pos: NDArray[np.int8]):
        """
        :param pos: game object's position
        """
        self.__pos = pos

    def update(self, *args):
        """
        Called at each step of the simulation
        """
        raise NotImplementedError

    def render(self, *args):
        """
        Shows the object on the screen
        """
        raise NotImplementedError