from nptyping import NDArray
import os

from .conventions import FIXED_BLOCK, BLOCK, CHARACTER, BOMB, FIRE

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class Renderer:
    def __init__(self, world: NDArray[bool], mode: str = None):
        if mode != 'draw' and mode != 'print' and mode is not None:
            raise ValueError('Invalid Renderer mode')

        self.__world = world
        self.__mode = mode

        if mode == 'draw':
            pygame.init()
            self.__width = pygame.display.Info().current_w
            self.__height = pygame.display.Info().current_h
            self.__display = pygame.display.set_mode((self.__width, self.__height),
                                                     pygame.RESIZABLE, 32)
            self.__clock = pygame.time.Clock()
        elif mode == 'print':
            self.__unicodes = {FIXED_BLOCK: '\u2588', BLOCK: '\u2591',
                               CHARACTER: '\u263A', BOMB: '\u2299', FIRE: '*'}
            self.__order = [CHARACTER, BOMB, FIRE, BLOCK, FIXED_BLOCK]

    def render(self, action: int = 0):
        if self.__mode == 'draw':
            self.__render_draw(action)
        elif self.__mode == 'print':
            self.__render_print()

    def __render_draw(self, action: int):
        pass

    def __render_print(self):
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
