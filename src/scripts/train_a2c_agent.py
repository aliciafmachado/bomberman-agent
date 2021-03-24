import argparse
import gym
from bomberman_rl.simulator.a2c_single_coach import A2CSingleCoach
from bomberman_rl.agent.a2c_agent import A2CAgent


# Parsing arguments
parser = argparse.ArgumentParser(description='Trains an Actor critic agent to destroy blocks alone')
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
    agent = A2CAgent.load(args.agent_pretrained_path, args.agent_pretrained_name)
else:
    agent = A2CAgent(env.observation_space.shape[0], env.observation_space.shape[1])

# Running train
coach = A2CSingleCoach(env, agent, "human",
                       show_each=500, nb_passes=5000, fps=10, gamma=0.99)
coach.run()

# Saving result
agent.save(args.agent_path, args.agent_name)
