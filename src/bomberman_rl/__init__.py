from gym.envs.registration import register

register(
    id='bomberman-default-v0',
    entry_point='bomberman_rl.envs:BombermanEnv',
    # Add keuword arguments like that
    # kwargs={'random_seed': 10}
)