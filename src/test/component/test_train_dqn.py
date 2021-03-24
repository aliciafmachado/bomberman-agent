import gym
from bomberman_rl.simulator.dqn_agent_single_coach import DQNAgentSingleCoach
from bomberman_rl.agent.dqn_agent import DQNAgent


env = gym.make("bomberman_rl:bomberman-default-v0")
plot_loss = False
plot_rewards = True
agent = DQNAgent(env.observation_space.shape, env.action_space.n, lr=1e-3, gamma=0.9)
coach = DQNAgentSingleCoach(env, agent, n_episodes=5000, show_each=50, max_steps=250,
                            exploration_init=0.9, exploration_end=0.1,
                            exploration_decay=400, plot_loss=plot_loss,
                            plot_rewards=plot_rewards)
coach.run()

agent.save('./', 'dqn_agent')
if plot_loss:
    coach.plot_loss()
if plot_rewards:
    coach.plot_rewards()
