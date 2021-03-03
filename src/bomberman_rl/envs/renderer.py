from nptyping import NDArray
from typing import Dict
import os

from .conventions import FIXED_BLOCK, BLOCK, CHARACTER, BOMB, FIRE, BLOCK_SIZE
from .sprites_factory import SpritesFactory

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class Renderer:
    """
    Representation of the game for us to see.
    """

    def __init__(self, world: NDArray[bool], game_objects: Dict = None,
                 mode: str = None):
        """
        Default constructor.
        Args:
            world: Game's matrix.
            game_objects: List of game objects in the world.
            mode: 'draw', 'print' or None, controls how the game will be rendered.
        """
        if mode != 'draw' and mode != 'print' and mode is not None:
            raise ValueError('Invalid Renderer mode')

        self.__world = world
        self.__game_objects = game_objects
        self.__mode = mode

        if mode == 'draw':
            pygame.init()
            self.__width = pygame.display.Info().current_w
            self.__height = pygame.display.Info().current_h
            self.__display = pygame.display.set_mode(
                (BLOCK_SIZE * world.shape[1], BLOCK_SIZE * world.shape[0]),
                pygame.SCALED, 32)
            self.__clock = pygame.time.Clock()

            self.__frames_per_step = 3
            self.__steps_per_sec = 2
            self.__grass_color = (16, 88, 48)
            self.__fixed_block_sprite = 'block2'
            self.__block_sprite = 'block1'
            self.__sprites_factory = SpritesFactory()
        elif mode == 'print':
            self.__unicodes = {FIXED_BLOCK: '\u2588', BLOCK: '\u2591',
                               CHARACTER: '\u263A', BOMB: '\u2299', FIRE: '*'}
            self.__order = [CHARACTER, BOMB, FIRE, BLOCK, FIXED_BLOCK]

    def render(self):
        """
        Draws or prints one game step.
        """
        if self.__mode == 'draw':
            self.__render_draw()
        elif self.__mode == 'print':
            self.__render_print()

    def __render_draw(self):
        """
        Draws the game 'frame_per_step' times, to make animations work.
        """
        for i in range(self.__frames_per_step):
            self.__display.fill(self.__grass_color)

            # Draw blocks
            for i in range(self.__world.shape[0]):
                for j in range(self.__world.shape[1]):
                    if self.__world[i, j, FIXED_BLOCK]:
                        self.__display.blit(
                            self.__sprites_factory[self.__fixed_block_sprite],
                            (BLOCK_SIZE * j, BLOCK_SIZE * i))
                    elif self.__world[i, j, BLOCK]:
                        self.__display.blit(self.__sprites_factory[self.__block_sprite],
                                            (BLOCK_SIZE * j, BLOCK_SIZE * i))

            # Draw game objects
            for list_obj in self.__game_objects.values():
                for obj in list_obj:
                    obj.render(self.__display, self.__sprites_factory,
                               self.__frames_per_step)

            pygame.display.flip()
            self.__clock.tick(self.__frames_per_step * self.__steps_per_sec)

    def __render_print(self):
        """
        Prints the 2D array of the game, giving priority to character, bomb, fire and
        blocks, in this order.
        """
        for i in range(self.__world.shape[0]):
            for j in range(self.__world.shape[1]):
                printed = False
                for o in self.__order:
                    if self.__world[i][j][o]:
                        print(self.__unicodes[o], end='')
                        printed = True
                        break
                if not printed:
                    print(end=' ')
            print()
        print()
