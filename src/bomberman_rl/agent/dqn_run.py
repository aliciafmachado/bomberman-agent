# Run main for DQNAgent

from bomberman_rl.memory.ReplayMemory import ReplayMemory
from bomberman_rl.memory.ReplayMemory import Simulation
from bomberman_rl.agent.DQNAgent import DQNAgent
import torchvision.transforms as transforms
import gym
import argparse
import torch

def main():
    # Parser
    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument("nb_episodes", type=int)

    # Optional arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--save_frequency", type=int, default=None)
    parser.add_argument("--retrain", type=str, default=None)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--memory_size", type=int, default=10000)

    args = parser.parse_args()

    # TODO: implement --save_model, --save_frequency, --retrain, --lr, etc
    # Args
    if args.seed != None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create a transform to convert the matrix to a tensor
    transform = transforms.ToTensor()

    # Create environment
    env = gym.make("bomberman_rl:bomberman-default-v0", display='print')

    # Create agent
    dqn_agent = DQNAgent(11, 13, 5)

    # Creating the memory
    memory = ReplayMemory(args.memory_size)

    # Create episodes and call training
    # TODO: create a function in this script to do this
    print("Beginning training . . . ")

    for i_episode in range(args.nb_episodes):
        state = env.reset()
        done = False
        loss = None

        while not done:
            # Select and perform an action
            action = dqn_agent.select_action(transform(state).unsqueeze(0).type(torch.FloatTensor).to(dqn_agent.device),
                i_episode)

            print("action: ", action.item())
            new_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=dqn_agent.device)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Next state
            state = next_state

            if len(memory) >= args.memory_size:
                sample = self.memory.sample(args.batch_size)
                batch = Simulation(*zip(*sample))
                loss = dqn_agent.train(batch, args.batch_size)

            if args.verbose == True:
                print('Episode[{}/{}], Loss: {:.4f}, Buffer state[{}/{}]'.format(
                        i_episode, args.nb_episodes,
                        loss, len(memory), args.memory_size))

        # Update or not the targetNet
        if i_episode % self.target_update == 0: 
            target_net.load_state_dict(qNet.state_dict())


    print("Ended training . . . ")

    # Evaluate agent
    print("Evaluating agent . . . ")

    print("Finished!")

if __name__ == '__main__':
    main()