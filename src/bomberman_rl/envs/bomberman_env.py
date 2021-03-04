import gym
import numpy as np
from typing import Optional, Tuple
from nptyping import NDArray

from .conventions import FIXED_BLOCK, BLOCK, CHARACTER, PLACE_BOMB, FIRE
from .game_objects.bomb import Bomb
from .game_objects.fire import Fire
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
        'available_board_sizes': [(11, 13), (5, 7), (5, 5)]
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
        self.initial_pos = np.array([1, 1], dtype=np.int8)
        self.bomb_duration = 3
        self.bomb_range = 3
        self.bomb_limit = 1

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

        self.game_objects = {'characters': [Character(self.initial_pos)],
                             'bombs': [],
                             'fires': []}

        self.renderer = Renderer(self.map, self.game_objects, display)
        self.__display = display

    def step(self, action: int):
        """
        :param action: next movement for the agent
        :return: observation, reward, done and info
        """
        if action == PLACE_BOMB and len(self.game_objects['bombs']) < self.bomb_limit:
            self.game_objects['bombs'].append(
                Bomb(self.game_objects['characters'][0].get_pos()))

        # Update bombs
        waiting = []
        for i in range(len(self.game_objects['bombs'])):
            if self.game_objects['bombs'][i].update(self.map):
                waiting.append(i)
            else:
                self.game_objects['fires'].append(
                    Fire(self.game_objects['bombs'][i].get_pos(), self.bomb_duration,
                         self.bomb_range, self.map, self.game_objects['characters'][0]))

        self.game_objects['bombs'] = [self.game_objects['bombs'][i] for i in waiting]

        # Update fires
        self.map[:, :, FIRE] = np.zeros(self.size)
        fires_indexes_to_keep = []
        for i in range(len(self.game_objects['fires'])):
            if self.game_objects['fires'][i].update():
                fires_indexes_to_keep.append(i)
        self.game_objects['fires'] = [self.game_objects['fires'][i] for i in fires_indexes_to_keep]

        # Update character
        died, reward = self.game_objects['characters'][0].update(action, self.map)
        done = not died

        # Get observation from map
        observation = np.copy(self.map)

        return observation, reward, done, {}

    def reset(self, new_map=False) -> NDArray[bool]:
        """
        :param new_map: if True generate new positions for the breakable blocks
        :return: map a numpy array which contains the description of the current state
        """
        if new_map:
            if self.custom_map:
                raise Exception("Can't create new map in custom map environment")
            self.map = self.__create_map_from_scratch()
            self.original_map = np.copy(self.map)
        else:
            self.map = np.copy(self.original_map)

        self.game_objects = {'characters': [Character(self.initial_pos)],
                             'bombs': [],
                             'fires': []}

        self.renderer = Renderer(self.map, self.game_objects, self.__display)

        return np.copy(self.map), False, 0, {}

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
