import gym
import numpy as np
from typing import Optional, Tuple, Dict, Union, List
from nptyping import NDArray

from .conventions import FIXED_BLOCK, BLOCK, PLACE_BOMB, FIRE, BOMB
from .game_objects.bomb import Bomb
from .game_objects.fire import Fire
from .game_objects.character import Character
from .game_objects.breaking_block import BreakingBlock
from .renderer import Renderer


class BombermanEnv(gym.Env):
    """
    Bomberman Environment:

    Possible actions: stop(0), left (1), right (2), up (3), down (4), place bomb (5)
    Map 3D Matrix: map[:, :, 0] = fixed blocks position (1)
                   map[:, :, 1] = normal blocks position         (S)
                   map[:, :, 2] = character position             (P)
                   map[:, :, 3] = bomb position                  (B)
                   map[:, :, 4] = fire position                  (F)
    """

    metadata = {
        'render.modes': ['human', 'stdout'],
        'available_board_sizes': [(11, 13), (5, 7), (5, 5)]
    }

    def __init__(self, size: Optional[Tuple[int, int]] = (11, 13), n_agents: int = 1,
                 custom_map: Optional[str] = None, random_seed: int = 42):
        """
        Environment constructor
        Args:
            size: optional map size, it will be ignored if a custom_map is given
            n_agents: number of bombermans in the game.
            custom_map: if given a path, it will load a custom map from a txt
            random_seed: numpy random seed for reproducibility
        """
        assert 1 <= n_agents <= 4

        self.custom_map = custom_map
        self.n_agents = n_agents
        self.bomb_duration = 6
        self.fire_duration = 4
        self.fire_range = 3

        np.random.seed(random_seed)

        # Map creation
        if custom_map:
            self.map = self.__create_map_from_file(custom_map)
            self.size = (self.map.shape[0], self.map.shape[1])
        else:
            if size not in BombermanEnv.metadata['available_board_sizes']:
                raise Exception("Map of size {} not implemented".format(size))
            self.size = size
            self.map = self.__create_map_from_scratch()
            self.original_map = np.copy(self.map)

        self.game_objects, self.character_layers = self.__init_game_objects()
        self.renderer = Renderer(self.map, self.character_layers, self.game_objects)

    def step(self, action: Union[int, List[int]]) -> \
            Tuple[Union[NDArray, List[NDArray]], Union[float, List[float]], bool, Dict]:
        """
        :param action: next movement for the agent(s)
        :return: observation(s), reward(s), done and info
        """
        if self.n_agents == 1:
            assert isinstance(action, int)
            action = [action]
        else:
            assert isinstance(action, list)

        for agent_idx in range(self.n_agents):
            agent = self.game_objects['characters'][agent_idx]
            if action[agent_idx] == PLACE_BOMB and agent.can_place_bomb():
                # Check if there is not a bomb there
                pos = self.game_objects['characters'][agent_idx].get_pos()
                x, y = pos
                if not self.map[x, y, BOMB]:
                    agent.place_bomb()
                    bomb = Bomb(pos, agent, self.bomb_duration)
                    self.game_objects['bombs'].append(bomb)

        # Update bombs
        bomb_pos = {
            tuple(bomb.get_pos()): bomb for bomb in self.game_objects['bombs']
        }
        set_all_bombs = set(self.game_objects['bombs'])
        set_remaining_bombs = set()
        while set_all_bombs:
            bomb = set_all_bombs.pop()
            if bomb.update(self.map):
                set_remaining_bombs.add(bomb)
            else:
                del bomb_pos[tuple(bomb.get_pos())]
                bomb.get_owner().bomb_exploded()

                # Add fire
                fire = Fire(bomb.get_pos(), self.fire_duration, self.fire_range, self.map,
                            bomb.get_owner())
                self.game_objects['fires'].append(fire)

                # Add breaking blocks
                for break_block_pos in fire.break_blocks:
                    self.game_objects['breaking_blocks'].append(
                        BreakingBlock(break_block_pos, self.fire_duration)
                    )

                # Explode bombs in the way
                for bomb_hit_pos in fire.bombs_hit:
                    bomb_hit = bomb_pos[tuple(bomb_hit_pos)]
                    if bomb_hit in set_remaining_bombs:
                        set_remaining_bombs.remove(bomb_hit)
                    bomb_hit.explode()
                    set_all_bombs.add(bomb_hit)

        self.game_objects['bombs'] = list(set_remaining_bombs)

        # Update fires
        self.map[:, :, FIRE] = np.zeros(self.size)
        fires_indexes_to_keep = []
        for i in range(len(self.game_objects['fires'])):
            if self.game_objects['fires'][i].update():
                fires_indexes_to_keep.append(i)
        self.game_objects['fires'] = [
            self.game_objects['fires'][i] for i in fires_indexes_to_keep]

        # Update breaking block
        breaking_blocks_idx_to_keep = []
        for i in range(len(self.game_objects['breaking_blocks'])):
            if self.game_objects['breaking_blocks'][i].update(self.map):
                breaking_blocks_idx_to_keep.append(i)
        self.game_objects['breaking_blocks'] = [
            self.game_objects['breaking_blocks'][i] for i in breaking_blocks_idx_to_keep]

        # Update characters
        done = 0
        rewards = [0 for _ in range(self.n_agents)]
        for agent_idx in range(self.n_agents):
            agent = self.game_objects['characters'][agent_idx]
            alive, reward = agent.update(action[agent_idx], self.map,
                                         self.character_layers[:, :, agent_idx])
            rewards[agent_idx] += reward
            done += 0 if alive else 1

            if agent.just_died() and self.n_agents > 1:
                # Give reward to correct agent
                for fire in self.game_objects['fires']:
                    if tuple(agent.get_pos()) in fire.get_occupied_tiles():
                        if not fire.get_owner().just_died():
                            rewards[fire.get_owner().get_idx()] += Character.kill_reward

        # Adjust for single agent
        done = (self.n_agents > 1 and done >= self.n_agents - 1) or \
               (self.n_agents == 1 and done > 0)
        rewards = rewards[0] if self.n_agents == 1 else rewards

        return self.__build_observation(), rewards, done, {}

    def reset(self, new_map: bool = False) -> Union[NDArray, List[NDArray]]:
        """
        :param new_map: if True generate new positions for the breakable blocks
        :return: map a numpy array which contains the description of the current state
        """
        if new_map:
            if self.custom_map:
                raise Exception("Can't create new map in custom map environment")
            self.map = self.__create_map_from_scratch()
            self.original_map = np.copy(self.map)
        else:
            self.map = np.copy(self.original_map)

        self.game_objects, self.character_layers = self.__init_game_objects()
        self.renderer.reset(self.map, self.character_layers, self.game_objects)

        return self.__build_observation()

    def render(self, mode: str = 'human', steps_per_sec: int = 2,
               debug_text: Optional[str] = None):
        """
        Renders or prints the step.
        Args:
            mode: 'human' or 'stdout' for pygame rendering or print.
            steps_per_sec: This controls the speed of the rendering if it is 'human'.
            debug_text: Additional text to be displayed.
        """
        if mode not in BombermanEnv.metadata['render.modes']:
            raise ValueError('Invalid render mode')

        self.renderer.render(mode, steps_per_sec, debug_text)

    def close(self):
        raise NotImplementedError

    def __init_game_objects(self):
        initial_pos = [np.array([1, 1], dtype=np.int8),
                       np.array([1, self.size[1] - 2], dtype=np.int8),
                       np.array([self.size[0] - 2, 1], dtype=np.int8),
                       np.array([self.size[0] - 2, self.size[1] - 2], dtype=np.int8)]
        game_objects = {
            'bombs': [],
            'fires': [],
            'characters': [],
            'breaking_blocks': []
        }
        character_layers = np.zeros((self.size[0], self.size[1], self.n_agents))
        for i in range(self.n_agents):
            game_objects['characters'].append(Character(initial_pos[i], i))
            character_layers[initial_pos[i][0], initial_pos[i][1], i] = 1

        return game_objects, character_layers

    def __build_observation(self):
        if self.n_agents > 1:
            obs = []
            for i in range(self.n_agents):
                st = np.stack((self.character_layers[:, :, i],
                              np.any(self.character_layers[
                                     :, :, np.arange(self.n_agents) != i], axis=2)),
                              axis=2)
                obs.append(np.concatenate((self.map, st), axis=2))

            return obs
        else:
            return np.append(self.map,
                             np.expand_dims(self.character_layers[:, :, 0], axis=2),
                             axis=2)

    def __create_map_from_file(self, custom_map: str) -> NDArray[bool]:
        # open file
        f = open(custom_map, "r")
        prov_map = []
        for line in f:
            l = line.split()
            prov_map.append(l)
        f.close()
        m, n = len(prov_map), len(prov_map[0])
        map = np.zeros((m, n, 4))
        for i in range(m):
            for j in range(n):
                value = prov_map[i][j]
                if value == '1':
                    map[i, j, FIXED_BLOCK] = 1
                elif value == 'S':
                    map[i, j, BLOCK] = 1
        return map

    def __create_map_from_scratch(self) -> NDArray[bool]:
        m, n = self.size
        map = np.zeros((m, n, 4))

        # Walls
        map[0, :, FIXED_BLOCK] = 1
        map[-1, :, FIXED_BLOCK] = 1
        map[:, 0, FIXED_BLOCK] = 1
        map[:, -1, FIXED_BLOCK] = 1

        # Fixed blocks
        rows = [2 * i for i in range(1, m // 2)]
        cols = [2 * i for i in range(1, n // 2)]
        tuples = [(r, c, FIXED_BLOCK) for r in rows for c in cols]
        for t in tuples:
            map[t] = 1

        # Soft blocks
        spawn_points = {(1, 1), (1, 2), (2, 1)}
        if self.n_agents > 1:
            spawn_points = spawn_points.union({(1, n - 2), (1, n - 3), (2, n - 2)})
        if self.n_agents > 2:
            spawn_points = spawn_points.union({(m - 2, 1), (m - 3, 1), (m - 2, 2)})
        if self.n_agents > 3:
            spawn_points = spawn_points.union({(m - 2, n - 2), (m - 2, n - 3), (m - 3, n - 2)})

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # avoid certain positions
                if (i, j) in spawn_points or map[i, j, FIXED_BLOCK]:
                    continue
                map[i, j, BLOCK] = np.random.rand() > 0.4
        return map
