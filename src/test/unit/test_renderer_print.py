import gym
import pytest
import unittest

from bomberman_rl.envs.Renderer import Renderer

class Case(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _pass_fixtures(self, capsys):
        self.capsys = capsys

    def test(self):
        expected_output = '█████████████\n' \
                          '█☺  ░░░   ░░█\n' \
                          '█ █░█ █░█░█ █\n' \
                          '█   ░░ ░   ░█\n' \
                          '█░█ █░█░█ █░█\n' \
                          '█  ░░⊙  ░░ ░█\n' \
                          '█ █░█ █░█ █░█\n' \
                          '█░ ░░░░***  █\n' \
                          '█ █ █ █░█ █ █\n' \
                          '█░ ░ ░░  ░░░█\n' \
                          '█████████████\n\n'

        env = gym.make("bomberman_rl:bomberman-default-v0")
        r = Renderer(env.map, mode='print')
        trues = [(5, 5, 3), (7, 7, 4), (7, 8, 4), (7, 9, 4)]
        for t in trues:
            env.map[t] = True
        r.render(0)

        # Capture print output
        captured = self.capsys.readouterr()
        self.assertEqual(expected_output, captured.out)
