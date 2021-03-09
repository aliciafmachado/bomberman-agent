from nptyping import NDArray
import numpy as np
import os

from bomberman_rl.envs.conventions import BLOCK_SIZE, BOMB
from bomberman_rl.envs.sprites_factory import SpritesFactory
from bomberman_rl.envs.game_objects.game_object import GameObject
from bomberman_rl.envs.game_objects.character import Character

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class Bomb(GameObject):
    def __init__(self, pos: NDArray[np.int8], owner: Character, force: int = 2, duration=6):
        super().__init__(pos)
        self.__owner = owner
        self.__force = force
        self.__duration = duration
        self.__animation_idx = 0
        self.__timer = -1

    def update(self, world: NDArray[bool]) -> bool:
        self.__timer += 1
        waiting = self.__timer < self.__duration
        world[self._pos[0], self._pos[1], BOMB] = waiting
        return waiting

    def render(self, display: pygame.display, sprites_factory: SpritesFactory,
               frames_per_step: int):
        screen_pos = self._pos.copy().astype(np.float)

        # Get correct animation sprite
        sprite_name = 'bomb'
        frame = self.__animation_idx % 4 + 1
        if frame == 4:
            frame = 2
        sprite_name += str(frame)  # Animation frame
        self.__animation_idx += 1

        # Draw sprite
        display.blit(sprites_factory[sprite_name], (screen_pos[1] * BLOCK_SIZE,
                                                    screen_pos[0] * BLOCK_SIZE))

    def explode(self):
        self.__timer = self.__duration

    def get_owner(self):
        return self.__owner
