import argparse
import gym
from bomberman_rl.simulator.dqn_agent_single_coach import DQNAgentSingleCoach
from bomberman_rl.agent.dqn_agent import DQNAgent


# Parsing arguments
parser = argparse.ArgumentParser(description='Trains a DQNAgent to destroy blocks alone')
parser.add_argument("--agent-name", help="agent file name after train", required=True)
parser.add_argument('--agent-path', help="path to agent file after train", default="./")
parser.add_argument("--agent-pretrained-name", help="name of pretrained agent to continue training", default=None)
parser.add_argument('--agent-pretrained-path', help="path of pretrained agent to continue training", default="./")
parser.add_argument('--environment', help="name of the environment to be used",
                    default="bomberman_rl:bomberman-default-v0")
parser.add_argument("--display", help="'human' or 'stdout'", default="human")
args = parser.parse_args()

env = gym.make(args.environment)

# Loading agent if needed or creating new one
agent = None
if args.agent_pretrained_name:
    agent = DQNAgent.load(args.agent_pretrained_path, args.agent_pretrained_name)
else:
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, lr=1e-3, gamma=0.9)

# Running train
coach = DQNAgentSingleCoach(env, agent, display=args.display, n_episodes=5000,
                            show_each=500, max_steps=250, exploration_init=0.9,
                            exploration_end=0.1, exploration_decay=400, plot_rewards=True)
coach.run()

# Saving result
agent.save(args.agent_path, args.agent_name)

coach.plot_rewards()
