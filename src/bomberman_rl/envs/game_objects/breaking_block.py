from nptyping import NDArray
import numpy as np
import os

from bomberman_rl.envs.conventions import BLOCK, BLOCK_SIZE
from bomberman_rl.envs.sprites_factory import SpritesFactory
from bomberman_rl.envs.game_objects.game_object import GameObject

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class BreakingBlock(GameObject):
    def __init__(self, pos: NDArray[np.int8], duration: int):
        super().__init__(pos)
        self.__duration = duration
        self.__animation_idx = 0
        self.__animation_list = [6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
        self.__timer = 0

    def update(self, world: NDArray[bool]) -> bool:
        """
        Update the block which is breaking.
        Args:
            world: The map's matrix.

        Returns:
            waiting: if the block is still there
        """
        self.__timer += 1
        waiting = self.__timer < self.__duration
        world[self._pos[0], self._pos[1], BLOCK] = waiting
        return waiting

    def render(self, display: pygame.display, sprites_factory: SpritesFactory,
               frames_per_step: int):
        # Get correct animation sprite
        sprite_name = 'block_destroyed'
        sprite_name += str(self.__animation_list[self.__animation_idx])  # Animation frame
        self.__animation_idx += 1

        # Draw sprite
        display.blit(sprites_factory[sprite_name], (self._pos[1] * BLOCK_SIZE,
                                                    self._pos[0] * BLOCK_SIZE))
