from gym.envs.registration import register

register(
    id='boberman-v0',
    entry_point='gym_foo.envs:BombermanEnv',
)