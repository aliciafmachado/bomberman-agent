import bomberman_rl
import pathlib
import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import pygame


class SpritesFactory:
    """
    Loads all of the images in the assets folder, and stores them in a dict.
    """
    def __init__(self):
        base_dir = (pathlib.Path(bomberman_rl.__file__).parent.parent / 'assets')
        files = base_dir.rglob('*.png')
        self.sprites = {}

        for f in files:
            self.sprites[f.name[:-4]] = pygame.image.load(f)

    def __getitem__(self, item):
        return self.sprites[item]
