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
            self.size = size
            self.map = self.__create_map_from_scratch()
        # bomb timer creation
        self.bombs = []
    
    def step(self, action):
        pass

    def reset(self):
        pass

    def __create_map_from_file(self, custom_map: str) -> NDArray[bool]:
        pass
    
    def __create_map_from_scratch(self) -> NDArray[bool]:
        m, n = self.size
        for i in range(m):
            for j in range(n):
                

    