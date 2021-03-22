from nptyping import NDArray
import numpy as np
import pygame
from typing import Dict, Tuple

from bomberman_rl.envs.game_objects.game_object import GameObject
from bomberman_rl.envs.game_objects.character import Character
from bomberman_rl.envs.sprites_factory import SpritesFactory

from bomberman_rl.envs.conventions import LEFT, RIGHT, DOWN, UP, \
    FIXED_BLOCK, BLOCK, BOMB, FIRE, BLOCK_SIZE, CENTER, HORIZONTAL, VERTICAL, END_LEFT, \
    END_RIGHT, END_DOWN, END_UP


class Fire(GameObject):
    """
    Represents a fire in the bomberman board
    """
    dir_dict = {LEFT: np.array([0, -1], dtype=np.int8),
                RIGHT: np.array([0, 1], dtype=np.int8),
                DOWN: np.array([1, 0], dtype=np.int8),
                UP: np.array([-1, 0], dtype=np.int8)}

    def __init__(self, pos: NDArray[np.int8], duration: int, tile_range: int,
                 world: NDArray, owner: Character):
        """
        :param pos: The center of the fire
        :param duration: for how many time
        :param tile_range:
        :param world:
        :param owner:
        """
        super().__init__(pos)
        self.__timer = -1
        self.__duration = duration
        self.__tiles = tile_range
        self.__world = world
        self.__owner = owner
        self.__occupied_tiles, self.break_blocks, self.bombs_hit = \
            self.__get_fire_coordinates()
        # animation assistance
        self.__animation_idx = 1
        self.__animation_end = False
        self.__animation_stop_4 = 0

    def update(self):
        """
        :return: If fire is still on the game.
        """
        self.__timer += 1
        waiting = self.__timer < self.__duration
        if waiting:
            self.__add_fire_to_map()
        return waiting
    
    def render(self, display: pygame.display, sprites_factory: SpritesFactory,
               frames_per_step: int):
        """
        Renders the fire.
        Args:
            display: Pygame display being used.
            sprites_factory: Class containing the dict of loaded sprites.
            frames_per_step: Number of frames per step in the game, to control the
                animation.
        """
        # load sprite based in position and frame
        if self.__animation_idx == 0:
            self.__animation_idx = 1
            self.__animation_end = False
            self.__animation_stop_4 = 0
        sprite_names = 'fire' + str(self.__animation_idx)
        # prepare next frame
        if self.__animation_stop_4 == 0 and self.__animation_end:
            self.__animation_stop_4 += 1
        elif self.__animation_stop_4 == 1 and self.__animation_end:
            self.__animation_stop_4 += 1
            self.__animation_idx += 1
        elif self.__animation_idx == 5:
            self.__animation_idx -= 1
            self.__animation_end = True
        elif self.__animation_idx < 5:
            if self.__animation_end:
                self.__animation_idx -= 1
            else:
                self.__animation_idx += 1
        # draw fire according to position
        for key in self.__occupied_tiles.keys():
            tp = self.__occupied_tiles[key]
            if tp == CENTER:
                sprite_name = sprite_names
            elif tp == HORIZONTAL:
                sprite_name = sprite_names + "_horizontal"
            elif tp == VERTICAL:
                sprite_name = sprite_names + "_vertical"
            elif tp == END_LEFT:
                sprite_name = sprite_names + "_left"
            elif tp == END_RIGHT:
                sprite_name = sprite_names + "_right"
            elif tp == END_UP:
                sprite_name = sprite_names + "_top"
            elif tp == END_DOWN:
                sprite_name = sprite_names + "_bot"
            else:
                raise Exception("Implement type = {}".format(tp))
            display.blit(sprites_factory[sprite_name],
                         (key[1] * BLOCK_SIZE, key[0] * BLOCK_SIZE))

    def get_occupied_tiles(self) -> Dict[Tuple[int, int], int]:
        return self.__occupied_tiles

    def get_owner(self) -> Character:
        return self.__owner

    def __add_fire_to_map(self):
        """
        Adds this fire to the map
        """
        for coord in self.__occupied_tiles.keys():
            self.__world[coord[0], coord[1], FIRE] = True
            self.__world[coord[0], coord[1], BLOCK] = False

    def __get_fire_coordinates(self):
        """
        Gets all of the coordinates that belong to the fire
        :return:
        """
        coordinates = {tuple(self._pos): CENTER}
        blocks_hit = []
        bombs_hit =[]
        # Grows the fire in each direction
        for dir in Fire.dir_dict:
            hit = False
            fire_pos = np.copy(self._pos)
            for i in range(self.__tiles):
                if hit:
                    break
                fire_pos = fire_pos + Fire.dir_dict[dir]
                # TODO elaborate these interactions
                if self.__world[fire_pos[0], fire_pos[1], FIXED_BLOCK]:
                    break
                if self.__world[fire_pos[0], fire_pos[1], BLOCK]:
                    hit = True
                    blocks_hit.append(fire_pos)
                    break
                if self.__world[fire_pos[0], fire_pos[1], BOMB]:
                    hit = True
                    bombs_hit.append(fire_pos)
                    break
                if dir == LEFT or dir == RIGHT:
                    if i == self.__tiles - 1:
                        coordinates[tuple(fire_pos)] = \
                            END_LEFT if dir == LEFT else END_RIGHT
                    else:
                        coordinates[tuple(fire_pos)] = HORIZONTAL
                elif dir == UP or dir == DOWN:
                    if i == self.__tiles - 1:
                        coordinates[tuple(fire_pos)] = END_UP if dir == UP else END_DOWN
                    else:
                        coordinates[tuple(fire_pos)] = VERTICAL

        return coordinates, blocks_hit, bombs_hit
