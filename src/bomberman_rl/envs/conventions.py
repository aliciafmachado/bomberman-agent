# Character's actions map to integer
STOP = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4
PLACE_BOMB = 5
N_ACTIONS = 6

# World matrix's layers indices
FIXED_BLOCK = 0
BLOCK = 1
BOMB = 2
FIRE = 3
CHARACTER = 4  # Hard coded in bomberman_env.__build_observation
ENEMIES = 5  # Hard coded in bomberman_env.__build_observation

# Block sprite size
BLOCK_SIZE = 16

# Fire position
CENTER = 0
HORIZONTAL = 1
VERTICAL = 2
END_LEFT = 3
END_RIGHT = 4
END_UP = 5
END_DOWN = 6
