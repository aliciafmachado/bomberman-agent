from nptyping import NDArray
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame

from .Action import Action


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

    def render(self, action: Action = Action.STOP):
        if self.__mode == 'draw':
            self.__render_draw(action)
        elif self.__mode == 'print':
            self.__render_print()

    def __render_draw(self, action: Action):
        pass

    def __render_print(self):
        unicodes = {'fixed_block': '\u2588', 'block': '\u2591', 'player': '\u263A',
                    'bomb': '\u2299', 'fire': '*'}
        # FIXME: standardize word_idx
        world_idx = {'fixed_block': 0, 'block': 1, 'player': 2, 'bomb': 3, 'fire': 4}
        order = ['player', 'bomb', 'fire', 'block', 'fixed_block']
        n, m = self.__world.shape[0:2]

        for i in range(self.__world.shape[0]):
            for j in range(self.__world.shape[1]):
                printed = False
                for o in order:
                    if self.__world[i][j][world_idx[o]]:
                        print(unicodes[o], end='')
                        printed = True
                        break
                if not printed:
                    print(end=' ')
            print()
        print()
