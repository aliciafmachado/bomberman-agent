from nptyping import NDArray
import numpy as np

from bomberman_rl.envs.game_objects.game_object import GameObject

from bomberman_rl.envs.conventions import LEFT, RIGHT, DOWN, UP, \
    FIXED_BLOCK, BLOCK, BOMB, FIRE, BLOCK_SIZE


class Fire(GameObject):
    """
    Represents a fire in the bomberman board
    """
    dir_dict = {LEFT: np.array([0, -1], dtype=np.int8),
                RIGHT: np.array([0, 1], dtype=np.int8),
                DOWN: np.array([1, 0], dtype=np.int8),
                UP: np.array([-1, 0], dtype=np.int8)}

    def __init__(self, pos: NDArray[np.int8], duration, tiles, world):
        """
        :param pos: The center of the fire
        :param duration: for how many time
        :param tiles:
        :param world:
        """
        super().__init__(pos)
        self.__timer = duration
        self.__tiles = tiles
        self.__world = world
        self.__occupied_tiles = self.__get_fire_coordinates()

    def update(self):
        """

        :param args:
        :return:
        """
        self.__timer -= 1
        self.__add__fire_to_map()
        return self.__timer

    def __add__fire_to_map(self):
        """
        Adds this fire to the map
        """
        for coord in self.__occupied_tiles:
            self.__world[coord[0], coord[1], FIRE] = True

    def __get_fire_coordinates(self):
        """
        Gets all of the coordinates that belong to the fire
        :return:
        """
        coordinates = [np.copy(self._pos)]

        # Grows the fire in each direction
        for dir in Fire.dir_dict:
            hit = False
            fire_pos = self._pos
            for i in range(self.__tiles):
                if hit:
                    break
                fire_pos = fire_pos + Fire.dir_dict[dir]
                # TODO elaborate these interactions
                if self.__world[fire_pos[0], fire_pos[1], FIXED_BLOCK]:
                    break
                if self.__world[fire_pos[0], fire_pos[1], BLOCK]:
                    hit = True
                coordinates.append(fire_pos)

        return coordinates
