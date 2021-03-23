import gym
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Union, List
from nptyping import NDArray

from .conventions import FIXED_BLOCK, BLOCK, PLACE_BOMB, FIRE, BOMB
from .game_objects.bomb import Bomb
from .game_objects.fire import Fire
from .game_objects.character import Character
from .game_objects.breaking_block import BreakingBlock
from .renderer import Renderer
from .reward import Reward


class BombermanEnv(gym.Env):
    """
    Bomberman Environment:

    Possible actions: stop(0), left (1), right (2), up (3), down (4), place bomb (5)
    Map 3D Matrix: map[:, :, 0] = fixed blocks position (1)
                   map[:, :, 1] = normal blocks position         (S)
                   map[:, :, 2] = bomb position                  (B)
                   map[:, :, 3] = fire position                  (F)
                   map[:, :, 4] = character position             (P)
    """

    metadata = {
        'render.modes': ['human', 'stdout'],
        'available_board_sizes': [(11, 13), (5, 7), (5, 5)]
    }

    def __init__(self, size: Optional[Tuple[int, int]] = (11, 13), centralized=False,
                 n_agents: int = 1, custom_map: Optional[str] = None,
                 random_seed: int = 42):
        """
        Environment constructor
        Args:
            size: optional map size, it will be ignored if a custom_map is given
            centralized: if the output matrix should be centralized in the player, or
                return an extra character layer with its position.
            n_agents: number of bombermans in the game.
            custom_map: if given a path, it will load a custom map from a txt
            random_seed: numpy random seed for reproducibility
        """
        assert 1 <= n_agents <= 4
        self.custom_map = custom_map
        self.centralized = centralized
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

        self.game_objects, self.character_layers, self.rewards = self.__init_game_objects()
        self.renderer = Renderer(self.map, self.character_layers, self.game_objects)

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(6)
        if centralized and n_agents == 1:
            shape = (size[0], size[1], 4)
        elif (centralized and n_agents > 1) or (not centralized and n_agents == 1):
            shape = (size[0], size[1], 5)
        else:
            shape = (size[0], size[1], 6)
        self.observation_space = gym.spaces.Box(0, 1, shape, np.bool)

    def step(self, action: Union[int, List[int]]) -> \
            Tuple[Union[NDArray, List[NDArray]], Union[float, List[float]], bool, Dict]:
        """
        :param action: next movement for the agent(s)
        :return: observation(s), reward(s), done and info
        """

        # reset rewards
        for r in self.rewards:
            r.reset_reward()

        if self.n_agents == 1:
            if isinstance(action, torch.Tensor):
                action = int(action)
            assert np.issubdtype(type(action), int)
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

                    # Reward if it will break blocks
                    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                    for d in dirs:
                        for r in range(self.fire_range):
                            tile = self.map[pos[0] + r * d[0], pos[1] + r * d[1]]
                            if tile[FIXED_BLOCK]:
                                break
                            elif tile[BLOCK] or tile[BOMB]:
                                self.rewards[agent_idx].add_will_break_block_reward()
                                break
            elif action[agent_idx] == PLACE_BOMB:
                self.rewards[agent_idx].add_illegal_movement_reward()

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
                    # give reward
                    self.rewards[fire.get_owner().get_idx()].add_broken_block_reward()

                # reward - no broken blocks
                if len(fire.break_blocks) == 0:
                    self.rewards[fire.get_owner().get_idx()].add_no_broken_block_reward()

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
        n_dead = 0
        for agent_idx in range(self.n_agents):
            agent = self.game_objects['characters'][agent_idx]
            self.rewards[agent_idx].calculate_reward(agent, action[agent_idx], self.map)
            alive = agent.update(action[agent_idx], self.map,
                                 self.character_layers[:, :, agent_idx])
            n_dead += 0 if alive else 1
            if agent.just_died() and self.n_agents > 1:
                # Give reward to correct agent
                for fire in self.game_objects['fires']:
                    if tuple(agent.get_pos()) in fire.get_occupied_tiles():
                        if not fire.get_owner().just_died():
                            self.rewards[fire.get_owner().get_idx()].add_kill_reward()

        reward_value = [self.rewards[i].get_reward() for i in range(self.n_agents)]

        # Adjust for single agent
        if self.n_agents > 1:
            done = n_dead >= self.n_agents - 1
        else:
            done = n_dead > 0 or not np.any(self.map[:, :, BLOCK])

        reward_value = reward_value[0] if self.n_agents == 1 else reward_value
        return self.__build_observation(), reward_value, done, {}

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

        self.game_objects, self.character_layers, self.rewards = self.__init_game_objects()
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
        character_layers = np.zeros((self.size[0], self.size[1], self.n_agents),
                                    dtype=bool)
        for i in range(self.n_agents):
            game_objects['characters'].append(Character(initial_pos[i], i))
            character_layers[initial_pos[i][0], initial_pos[i][1], i] = 1

        return game_objects, character_layers, [Reward() for i in range(self.n_agents)]

    def __build_observation(self):
        def centralize(matrix, pos):
            i0 = max(0, pos[0] - matrix.shape[0] // 2)
            j0 = max(0, pos[1] - matrix.shape[1] // 2)
            i1 = min(pos[0] + matrix.shape[0] // 2 + 1, matrix.shape[0])
            j1 = min(pos[1] + matrix.shape[1] // 2 + 1, matrix.shape[1])
            new_i0 = max(0, matrix.shape[0] // 2 - pos[0])
            new_j0 = max(0, matrix.shape[1] // 2 - pos[1])
            new_i1 = new_i0 + (i1 - i0)
            new_j1 = new_j0 + (j1 - j0)

            new_matrix = np.zeros_like(matrix, dtype=bool)
            new_matrix[new_i0:new_i1, new_j0:new_j1, :] = matrix[i0:i1, j0:j1, :]
            return new_matrix

        if self.n_agents > 1:
            obs = []
            for i in range(self.n_agents):
                character_layer = self.character_layers[:, :, i]
                enemies_layer = np.any(self.character_layers[
                                       :, :, np.arange(self.n_agents) != i], axis=2)

                if self.centralized:
                    matrix = np.append(self.map, np.expand_dims(enemies_layer, axis=2),
                                       axis=2)
                    pos = self.game_objects['characters'][i].get_pos()
                    obs.append(centralize(matrix, pos))
                else:
                    st = np.stack((character_layer, enemies_layer), axis=2)
                    obs.append(np.concatenate((self.map, st), axis=2))

            return obs
        else:
            if self.centralized:
                pos = self.game_objects['characters'][0].get_pos()
                return centralize(self.map, pos)
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
        map = np.zeros((m, n, 4), dtype=bool)
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
        map = np.zeros((m, n, 4), dtype=bool)

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
            spawn_points = spawn_points.union(
                {(m - 2, n - 2), (m - 2, n - 3), (m - 3, n - 2)})

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # avoid certain positions
                if (i, j) in spawn_points or map[i, j, FIXED_BLOCK]:
                    continue
                map[i, j, BLOCK] = np.random.rand() > 0.4
        return map
