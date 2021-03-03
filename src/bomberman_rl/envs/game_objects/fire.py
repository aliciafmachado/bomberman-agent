from nptyping import NDArray
import numpy as np
import pygame
import gym
from collections import defaultdict

from bomberman_rl.envs.game_objects.game_object import GameObject
from bomberman_rl.envs.sprites_factory import SpritesFactory

from bomberman_rl.envs.conventions import CENTER, LEFT, RIGHT, DOWN, UP, \
    FIXED_BLOCK, BLOCK, BOMB, FIRE, BLOCK_SIZE, CENTER, HORIZONTAL, VERTICAL, \
    END_LEFT, END_RIGHT, END_UP, END_DOWN

class Fire(GameObject):
    """
    Represents a fire in the bomberman board
    """
    dir_dict = {LEFT: np.array([0, -1], dtype=np.int8),
                RIGHT: np.array([0, 1], dtype=np.int8),
                DOWN: np.array([1, 0], dtype=np.int8),
                UP: np.array([-1, 0], dtype=np.int8)}

    def __init__(self, pos: NDArray[np.int8], duration, tiles, world, owner):
        """
        :param pos: The center of the fire
        :param duration: for how many time
        :param tiles:
        :param world:
        :param owner:
        """
        super().__init__(pos)
        self.__timer = duration
        self.__tiles = tiles
        self.__world = world
        self.__owner = owner
        self.__reward_given = False
        self.__occupied_tiles, self.break_blocks = self.__get_fire_coordinates()
        # animation assistance
        self.__animation_idx = 1
        self.__animation_end = False
        self.__animation_stop_4 = 0

    def update(self):
        """
        :param args:
        :return:
        """
        self.__timer -= 1
        self.__add__fire_to_map()
        return self.__timer > 0
    
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
        screen_pos = self._pos.copy().astype(np.float)
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
            screen_pos = np.array(key).copy().astype(np.float)
            tp = self.__occupied_tiles[key]
            if tp == CENTER:
                sprite_name = sprite_names
            elif tp == HORIZONTAL:
                sprite_name = sprite_names + "_horizontal"
            elif tp == VERTICAL:
                sprite_name = sprite_names + "_vertical"
            else:
                raise Exception("Implement type = {}".format(tp))
            display.blit(sprites_factory[sprite_name], (screen_pos[1] * BLOCK_SIZE,
                                            screen_pos[0] * BLOCK_SIZE))

    def __add__fire_to_map(self):
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
        coordinates = {}
        coordinates[tuple(np.copy(self._pos))] = CENTER
        blocks_hit = []
        # Grows the fire in each direction
        for dir in Fire.dir_dict:
            hit = False
            fire_pos = self._pos
            for _ in range(self.__tiles):
                if hit:
                    break
                fire_pos = fire_pos + Fire.dir_dict[dir]
                # TODO elaborate these interactions
                if self.__world[fire_pos[0], fire_pos[1], FIXED_BLOCK]:
                    break
                if self.__world[fire_pos[0], fire_pos[1], BLOCK]:
                    hit = True
                    blocks_hit.append(fire_pos)
                if dir == LEFT or dir == RIGHT:
                    coordinates[tuple(np.copy(fire_pos))] = HORIZONTAL
                elif dir == UP or dir == DOWN:
                    coordinates[tuple(np.copy(fire_pos))] = VERTICAL
            # Adding reward to player
            if not self.__reward_given and hit:
                self.__owner.break_block()

        self.__reward_given = True

        return coordinates, blocks_hit
