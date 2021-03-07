from nptyping import NDArray
from typing import Tuple
import numpy as np
import os

from bomberman_rl.envs.conventions import STOP, PLACE_BOMB, LEFT, RIGHT, DOWN, UP, \
    FIXED_BLOCK, BLOCK, BOMB, FIRE, BLOCK_SIZE
from bomberman_rl.envs.sprites_factory import SpritesFactory
from bomberman_rl.envs.game_objects.game_object import GameObject

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class Character(GameObject):
    """
    Contains character's position and movement info, its collision with fire and blocks,
    and the animation rendering.
    """
    dir_dict = {STOP: np.zeros(2, dtype=np.int8),
                PLACE_BOMB: np.zeros(2, dtype=np.int8),
                LEFT: np.array([0, -1], dtype=np.int8),
                RIGHT: np.array([0, 1], dtype=np.int8),
                DOWN: np.array([1, 0], dtype=np.int8),
                UP: np.array([-1, 0], dtype=np.int8)}
    bomb_limit = 1
    alive_reward = 0
    block_break_reward = 100
    dead_reward = -10
    kill_reward = 1000

    def __init__(self, pos: NDArray[np.int8], idx: int):
        """
        Default constructor.
        Args:
            pos: Tile i, j in the game map.
            idx: Agent's index
        """
        super().__init__(pos)
        self.__idx = idx
        self.__dir = self.dir_dict[DOWN]
        self.__stopped = True
        self.__animation_idx = 0
        self.__dead = False
        self.__just_died = False
        self.__block_break_cumulative_reward = 0
        self.__animation_idx_death = 0
        self.__placed_bombs = 0

    def update(self, action: int, world: NDArray[bool], world_layer: NDArray[bool]) -> \
            Tuple[bool, int]:
        """
        Updates the character's position according to its action and to the world.
        Args:
            action: One of the actions defined in conventions.py.
            world: The map's matrix.
            world_layer: The character's layer in the world.

        Returns:
            alive: True if the character is still alive.
            reward: The current reward
        """

        # Remove player from map
        world_layer[self._pos[0], self._pos[1]] = False
        self.__stopped = True

        if self.__dead:
            return False, self.__get_reward()

        # If player needs to move and it's able to move
        if action not in (STOP, PLACE_BOMB):
            # Get direction caused by the action
            self.__dir = self.dir_dict[action]

            next_pos = self._pos + self.__dir

            # Check for collision with blocks/bomb
            candidate_pos_objects = world[next_pos[0], next_pos[1]]
            obstacles = FIXED_BLOCK, BLOCK, BOMB
            if not any([candidate_pos_objects[o] for o in obstacles]):
                self.__stopped = False
                self._pos = next_pos

        # Check if there's fire in the next position
        if world[self._pos[0], self._pos[1], FIRE]:
            self.__dead = True
            self.__just_died = True
            return False, self.__get_reward()

        world_layer[self._pos[0], self._pos[1]] = True

        return True, self.__get_reward()

    def render(self, display: pygame.display, sprites_factory: SpritesFactory,
               frames_per_step: int):
        """
        Renders the character using a walking animation.
        Args:
            display: Pygame display being used.
            sprites_factory: Class containing the dict of loaded sprites.
            frames_per_step: Number of frames per step in the game, to control the
                animation.
        """
        screen_pos = self._pos.copy().astype(np.float)
        screen_pos[0] -= 0.25  # Character sprite is 25% taller than blocks
        sprite_name = 'bomberman' + str(self.__idx + 1) + '_'

        # In case of death
        if self.__just_died:
            self.__just_died = False
            if not self.__stopped:
                screen_pos -= self.__dir * (
                (1 - (self.__animation_idx % frames_per_step + 1) / frames_per_step))
            sprite_name += 'die' + str(self.__animation_idx_death % 3 + 1)
            self.__animation_idx_death += 1
            display.blit(sprites_factory[sprite_name], (screen_pos[1] * BLOCK_SIZE,
                                                        screen_pos[0] * BLOCK_SIZE))
            return
        elif self.__dead:
            return

        # Get correct animation frame
        if self.__dir[0] == -1:
            sprite_name += 'up'
        elif self.__dir[0] == 1:
            sprite_name += 'down'
        elif self.__dir[1] == -1:
            sprite_name += 'left'
        else:
            sprite_name += 'right'

        # Put character in the correct position on screen (between two blocks)
        if not self.__stopped:
            screen_pos -= self.__dir * (frames_per_step - 1 - (
                    self.__animation_idx % frames_per_step)) / frames_per_step
            if self.__animation_idx % 4 == 0:
                sprite_name += '1'
            elif self.__animation_idx % 4 == 2:
                sprite_name += '2'
            self.__animation_idx += 1

        # Draw sprite
        display.blit(sprites_factory[sprite_name], (screen_pos[1] * BLOCK_SIZE,
                                                    screen_pos[0] * BLOCK_SIZE))

    def get_idx(self) -> int:
        return self.__idx

    def break_block(self):
        self.__block_break_cumulative_reward += Character.block_break_reward

    def can_place_bomb(self) -> bool:
        return self.__placed_bombs < Character.bomb_limit

    def place_bomb(self):
        self.__placed_bombs += 1

    def bomb_exploded(self):
        self.__placed_bombs -= 1

    def just_died(self) -> bool:
        return self.__just_died

    def __get_reward(self):
        if self.__dead:
            self.__block_break_cumulative_reward = 0
            return Character.dead_reward
        else:
            block_reward = self.__block_break_cumulative_reward
            self.__block_break_cumulative_reward = 0
            return Character.alive_reward + block_reward
