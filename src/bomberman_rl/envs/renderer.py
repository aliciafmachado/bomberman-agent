from nptyping import NDArray
import numpy as np
from typing import Dict, Optional
import os

from .conventions import FIXED_BLOCK, BLOCK, CHARACTER, BOMB, FIRE, BLOCK_SIZE
from .sprites_factory import SpritesFactory

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class Renderer:
    """
    Representation of the game for us to see.
    """
    sprites_factory = None
    frames_per_step = 3
    grass_color = (16, 88, 48)
    block_sprite = 'block1'
    fixed_block_sprite = 'block2'
    unicodes = {FIXED_BLOCK: '\u2588', BLOCK: '\u2591',
                CHARACTER: '\u263A', BOMB: '\u2299', FIRE: '*'}
    print_order = [CHARACTER, BOMB, FIRE, BLOCK, FIXED_BLOCK]

    def __init__(self, world: NDArray[bool], character_layers: NDArray[bool],
                 game_objects: Optional[Dict] = None):
        """
        Default constructor.
        Args:
            world: Game's matrix.
            character_layers: Game's characters matrix.
            game_objects: List of game objects in the world.
        """

        self.__world = world
        self.__character_layers = character_layers
        self.__game_objects = game_objects
        self.__first_time = True
        self.__mode = None

        pygame.init()
        pygame.font.init()
        self.__clock = pygame.time.Clock()
        self.__font = pygame.font.SysFont('arial', 10)
        self.__width = None
        self.__height = None
        self.__display = None

    def render(self, mode: str, steps_per_sec: int, debug_text: Optional[str]):
        """
        Draws or prints one game step.
        @param mode: 'human' or 'stdout'
        @param steps_per_sec: Controls the speed of the game.
        @param debug_text: Additional text to be displayed.
        """
        if self.__first_time:
            self.__first_time = False
            self.__mode = mode

            if mode == 'human':
                self.__width = pygame.display.Info().current_w
                self.__height = pygame.display.Info().current_h
                self.__display = pygame.display.set_mode((
                    BLOCK_SIZE * self.__world.shape[1],
                    BLOCK_SIZE * self.__world.shape[0]), pygame.SCALED, 32)

                if Renderer.sprites_factory is None:
                    Renderer.sprites_factory = SpritesFactory()

        if self.__mode == 'human':
            self.__render_human(steps_per_sec, debug_text)
        elif self.__mode == 'stdout':
            self.__render_stdout(steps_per_sec, debug_text)

    def reset(self, world: NDArray[bool], character_layers: NDArray[bool],
              game_objects: Optional[Dict] = None):
        """
        Resets the window
        Args:
            world: Game's matrix.
            character_layers: Game's characters matrix.
            game_objects: List of game objects in the world.
        """
        if self.__mode == "human":
            del self.__font  # Fixes segfault https://github.com/renpy/pygame_sdl2/issues/112
            pygame.quit()
        self.__init__(world, character_layers, game_objects)

    def __render_human(self, steps_per_sec: int, debug_text: Optional[str]):
        """
        Draws the game 'frame_per_step' times, to make animations work.
        @param steps_per_sec: Controls the speed of the game.
        @param debug_text: Additional text to be displayed.
        """
        for i in range(Renderer.frames_per_step):
            self.__display.fill(Renderer.grass_color)

            # Draw blocks
            for i in range(self.__world.shape[0]):
                for j in range(self.__world.shape[1]):
                    if self.__world[i, j, FIXED_BLOCK]:
                        self.__display.blit(
                            Renderer.sprites_factory[Renderer.fixed_block_sprite],
                            (BLOCK_SIZE * j, BLOCK_SIZE * i))
                    elif self.__world[i, j, BLOCK]:
                        self.__display.blit(
                            Renderer.sprites_factory[Renderer.block_sprite],
                            (BLOCK_SIZE * j, BLOCK_SIZE * i))

            # Draw game objects
            for list_obj in self.__game_objects.values():
                for obj in list_obj:
                    obj.render(self.__display, Renderer.sprites_factory,
                               Renderer.frames_per_step)

            # Display text
            if debug_text is not None:
                text_surface = self.__font.render(debug_text, False, (255, 255, 255))
                self.__display.blit(text_surface, (1, 1))

            pygame.display.flip()
            self.__clock.tick(Renderer.frames_per_step * steps_per_sec)

    def __render_stdout(self, steps_per_sec: int, debug_text: Optional[str]):
        """
        Prints the 2D array of the game, giving priority to character, bomb, fire and
        blocks, in this order.
        @param steps_per_sec: Controls the speed of the game.
        @param debug_text: Additional text to be printed.
        """
        for i in range(self.__world.shape[0]):
            for j in range(self.__world.shape[1]):
                printed = False
                for o in Renderer.print_order:
                    pos = np.any(self.__character_layers[i, j]) if o == CHARACTER \
                        else self.__world[i][j][o]
                    if pos:
                        print(Renderer.unicodes[o], end='')
                        printed = True
                        break
                if not printed:
                    print(end=' ')
            print()

        if debug_text is not None:
            print(debug_text)

        print()
        self.__clock.tick(steps_per_sec)
