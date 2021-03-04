from gym.envs.registration import register

register(
    id='bomberman-default-v0',
    entry_point='bomberman_rl.envs:BombermanEnv'
)

register(
    id='bomberman-small-v0',
    entry_point='bomberman_rl.envs:BombermanEnv',
    kwargs={'size': (5, 7)}
)
register(
    id='bomberman-minimal-v0',
    entry_point='bomberman_rl.envs:BombermanEnv',
    kwargs={'size': (5, 5)}
)

