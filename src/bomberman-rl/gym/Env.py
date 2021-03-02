import gym
from gym import spaces
import numpy as np
from typing import List, Optional, Tuple
from nptyping import NDArray

class Env(gym.Env):
    """
    Bomberman Environment:

    Possible actions: left (0), up (1), right (2), down (3), put bomb (4)
    Map 3D Matrix: map[:, :, 0] = indestructible blocks position (1)
                   map[:, :, 1] = soft blocks position           (S)
                   map[:, :, 2] = player position                (P)
                   map[:, :, 3] = bomb position                  (B)
                   map[:, :, 4] = fire position                  (F)
    """
    def __init__(self, size: Optional(Tuple[int, int]) = (11, 13),
                       display: bool = False, 
                       custom_map: Optional[str]=None, 
                       random_seed: int = 42):
        """
        Environment constructor
        Args:
            size: optional map size, it will be ignored if a custom_map is given
            display: display the board
            custom_map: if given a path, it will load a custom map from a txt
            random_seed: numpy random seed for reproducibility
        """
        np.random.seed(random_seed)
        # map creation
        if custom_map:
            self.map = self.__create_map_from_file(custom_map)
            self.size = (self.map.shape[0], self.map.shape[1])
        else:
            # TODO
            # implement a more generalist map construction
            if size != (11, 13) or size != (5, 7):
                raise Exception("Map of size {} not implemented".format(size))
            self.size = size
            self.map = self.__create_map_from_scratch()
        # bomb timer creation
        self.bombs = []
    
    def step(self, action):
        pass

    def reset(self):
        pass

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
                    map[i, j, 0] = 1
                elif value == 'S':
                    map[i, j, 1] = 1
                elif value == 'P':
                    map[i, j, 2] = 1
        return map
        
    def __create_map_from_scratch(self) -> NDArray[bool]:
        m, n = self.size
        map = np.zeros((m, n, 5))
        # walls
        map[0, :, 0] = 1
        map[-1, :, 0] = 1
        map[:, 0, 0] = 1
        map[:, -1, 0] = 1
        # fixed blocks
        rows = [2*i for i in range(1, m//2)]
        cols = [2*i for i in range(1, n//2)]
        tuples = [(r, c, 0) for r in rows for c in cols]
        for t in tuples:
            map[t] = 1
        # player position
        map[1, 1, 2] = 1
        # soft blocks
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # avoid certain positions
                if (i, j) in [(1, 1), (1, 2), (2, 1)] or map[i, j, 0] == 1:
                    continue
                map[i, j, 1] = np.random.rand() > 0.4
        return map
    