import gym
from gym import spaces
import numpy as np
from typing import Optional, Tuple
from nptyping import NDArray

from .conventions import FIXED_BLOCK, BLOCK, CHARACTER
from .game_objects.character import Character
from .renderer import Renderer


class BombermanEnv(gym.Env):
    """
    Bomberman Environment:

    Possible actions: stop(0), left (1), right (2), up (3), down (4), place bomb (5)
    Map 3D Matrix: map[:, :, 0] = fixed blocks position (1)
                   map[:, :, 1] = normal blocks position         (S)
                   map[:, :, 2] = character position             (P)
                   map[:, :, 3] = bomb position                  (B)
                   map[:, :, 4] = fire position                  (F)
    """

    metadata = {
        'render.modes': ['human', 'stdout'],
        'available_board_sizes': [(11, 13), (5, 7)]
    }

    def __init__(self, size: Optional[Tuple[int, int]] = (11, 13),
                 display: str = None,
                 custom_map: Optional[str] = None,
                 random_seed: int = 42):
        """
        Environment constructor
        Args:
            size: optional map size, it will be ignored if a custom_map is given
            display: display the board
            custom_map: if given a path, it will load a custom map from a txt
            random_seed: numpy random seed for reproducibility
        """
        self.custom_map = custom_map
        self.initial_pos = [1, 1]

        np.random.seed(random_seed)
        # Map creation
        if custom_map:
            self.map = self.__create_map_from_file(custom_map)
            self.size = (self.map.shape[0], self.map.shape[1])
        else:
            if size not in BombermanEnv.metadata['available_board_sizes']:
                raise Exception("Map of size {} not implemented".format(size))
            self.size = size
            self.map = self.__create_map_from_scratch()
            self.original_map = np.copy(self.map)

        # bomb timer creation
        # self.bombs = []

        self.character = Character(np.array(self.initial_pos, dtype=np.int8))
        self.game_objects = [self.character]

        self.renderer = Renderer(self.map, self.game_objects, display)

    def step(self, action: int):
        """
        :param action: next movement for the agent
        :return: observation, reward, done and info
        """
        self.character.update(action, self.map)

        # Stepping all of the bombs (start and end explosions)

        # Getting the next observation
        old_pos = self.character.get_pos()
        done = not self.character.update(action, self.map)
        new_pos = self.character.get_pos()

        self.map[old_pos[0], old_pos[1], CHARACTER] = 0
        self.map[new_pos[0], new_pos[1], CHARACTER] = 1
        observation = np.copy(self.map)

        # Placing bomb if needed and if it's possible

        # Getting reward

        # return observation, reward, done, info

    def reset(self, new_map=False) -> NDArray[bool]:
        """
        :param new_map: if True generate new positions for the breakable blocks
        :return: map a numpy array which contains the description of the current state
        """
        if new_map:
            if self.custom_map:
                raise Exception("Can't cresate new map in custom map environment")
            self.map = self.__create_map_from_scratch()
            self.original_map = np.copy(self.map)
        else:
            self.map = np.copy(self.original_map)

        self.character.set_pos(self.initial_pos)

        return np.copy(self.map)

    def render(self, mode='human'):
        self.renderer.render()

    def close(self):
        raise NotImplementedError

    def __create_map_from_file(self, custom_map: str) -> NDArray[bool]:
        # open file
        f = open(custom_map, "r")
        prov_map = []
        for line in f:
            l = line.split()
            prov_map.append(l)
        f.close()
        m, n = len(prov_map), len(prov_map[0])
        map = np.zeros((m, n, 5))
        for i in range(m):
            for j in range(n):
                value = prov_map[i][j]
                if value == '1':
                    map[i, j, FIXED_BLOCK] = 1
                elif value == 'S':
                    map[i, j, BLOCK] = 1
                elif value == 'P':
                    map[i, j, CHARACTER] = 1
        return map

    def __create_map_from_scratch(self) -> NDArray[bool]:
        m, n = self.size
        map = np.zeros((m, n, 5))
        # walls
        map[0, :, FIXED_BLOCK] = 1
        map[-1, :, FIXED_BLOCK] = 1
        map[:, 0, FIXED_BLOCK] = 1
        map[:, -1, FIXED_BLOCK] = 1
        # fixed blocks
        rows = [2 * i for i in range(1, m // 2)]
        cols = [2 * i for i in range(1, n // 2)]
        tuples = [(r, c, FIXED_BLOCK) for r in rows for c in cols]
        for t in tuples:
            map[t] = 1
        # player position
        map[self.initial_pos[0], self.initial_pos[1], CHARACTER] = 1
        # soft blocks
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # avoid certain positions
                if (i, j) in [(1, 1), (1, 2), (2, 1)] or map[i, j, FIXED_BLOCK]:
                    continue
                map[i, j, BLOCK] = np.random.rand() > 0.4
        return map