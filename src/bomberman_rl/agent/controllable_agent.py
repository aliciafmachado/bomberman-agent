import os
import sys

from .agent import Agent
from bomberman_rl.envs.conventions import UP, DOWN, RIGHT, LEFT, STOP, PLACE_BOMB

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class ControllableAgent(Agent):
    """
    An agent that is controlled by the keyboard.
    """

    controls = {pygame.K_w: UP, pygame.K_s: DOWN, pygame.K_a: LEFT,
                pygame.K_d: RIGHT, pygame.K_SPACE: PLACE_BOMB}

    def __init__(self):
        super().__init__()
        self.__movement = STOP
        self.__place_bomb = False

    def choose_action(self) -> int:
        """
        Looks at pygame events and chooses the correct action.

        Returns:
            An action, as determined in conventions.py.
        """

        if self.__place_bomb:
            self.__place_bomb = False

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in ControllableAgent.controls:
                    new_action = ControllableAgent.controls[event.key]
                    if new_action != PLACE_BOMB:
                        self.__movement = new_action

            elif event.type == pygame.KEYUP:
                if event.key in ControllableAgent.controls:
                    if ControllableAgent.controls[event.key] == PLACE_BOMB:
                        self.__place_bomb = True
                    else:
                        self.__movement = STOP

            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        return PLACE_BOMB if self.__place_bomb else self.__movement
