from .conventions import STOP, PLACE_BOMB, LEFT, RIGHT, DOWN, UP, \
    FIXED_BLOCK, BLOCK, BOMB, FIRE, BLOCK_SIZE

from .game_objects.character import Character

import pprint

class Reward(object):

    # reward values
    rewards = {
        'visited_cell': 0.0,
        'new_cell': 0.04,
        'illegal_movement': -0.2,
        'no_obstacles_destroyed': -0.05,
        'will_destroy_block': 0.1,
        'destroy_block': 1,
        'death': -0.5,
        'kill_reward': 1,
        'place_bomb': 0.0,
    }

    def __init__(self):
        self._visited_cells = set()
        self._reward = 0
        self._log = {}
        self._total_reward = 0

    def calculate_reward(self, agent, action, world):
        self._log['agent_idx'] = agent.get_idx()
        self._log['action'] = action
        
        # get agent position
        position = agent.get_pos()
        self._visited_cells.add(tuple(position))

        dirr = agent.dir_dict[action]

        next_pos = position + dirr
        if action not in (STOP, PLACE_BOMB):
            # Get direction caused by the action
            candidate_pos_objects = world[next_pos[0], next_pos[1]]
            obstacles = FIXED_BLOCK, BLOCK, BOMB
            # Check for collision with blocks/bomb
            if not any([candidate_pos_objects[o] for o in obstacles]):
                if tuple(next_pos) in self._visited_cells:
                    self._reward += Reward.rewards['visited_cell']
                    self._log['rewards'].append({'visited_cells': Reward.rewards['visited_cell']})
                else:
                    self._reward += Reward.rewards['new_cell']
                    self._log['rewards'].append({'new_cell': Reward.rewards['new_cell']})
            else:
                self._reward += Reward.rewards['illegal_movement']
                self._log['rewards'].append({'illegal_movement': Reward.rewards['illegal_movement']})
        # check if position more bombs that allowed
        if action == PLACE_BOMB:
            self._reward += Reward.rewards['place_bomb']
            self._log['rewards'].append({'place_bomb': Reward.rewards['place_bomb']})                
        # Check if there's fire in the next position
        if world[next_pos[0], next_pos[1], FIRE]:
            self._reward += Reward.rewards['death']
            self._log['rewards'].append({'death': Reward.rewards['death']})

    def add_illegal_movement_reward(self):
        self._reward += Reward.rewards['illegal_movement']
        self._log['rewards'].append({'illegal_movement': Reward.rewards['illegal_movement']})

    def add_kill_reward(self):
        self._reward += Reward.rewards['kill_reward']
        self._log['rewards'].append({'kill_reward': Reward.rewards['kill_reward']})

    def add_will_break_block_reward(self):
        self._reward = Reward.rewards['will_destroy_block']
        self._log['rewards'].append(
            {'will_destroy_block': Reward.rewards['will_destroy_block']})

    def add_broken_block_reward(self):
        self._reward += Reward.rewards['destroy_block']
        self._log['rewards'].append({'destroy_block': Reward.rewards['destroy_block']})

    def add_no_broken_block_reward(self):
        self._reward += Reward.rewards['no_obstacles_destroyed']
        self._log['rewards'].append({'no_obstacles_destroyed': Reward.rewards['no_obstacles_destroyed']})

    def get_reward(self):
        return self._reward

    def reset_reward(self):
        self._total_reward += self._reward
        self._reward = 0
        self._log = {}
        self._log['rewards'] = []

    def showReward(self):
        pp = pprint.PrettyPrinter()
        pp.pprint(self._log)