import ppo as method
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import veins_gym

import argparse
import os

# Define the folder to store checkpoints
checkpoint_folder = 'model_checkpoints'

# Create the folder if it doesn't exist
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)

gym.register(
    id="serpentine",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../../scenario",
    },
)

gym.register(
    id="lysegeven",
    entry_point="veins_gym:VeinsEnv",
    kwargs={
        "scenario_dir": "../../scenario3",
    },
)

class FedPPO:

    def __init__(self, envs, device, **kwargs):
        
        self.num_clients = len(envs)
        self.kwargs = kwargs

        # initialize PPO objects
        self.ppo_objs = [method.PPO(env, device, **self.kwargs) for env in envs]

        #self.obs_dim, self.act_dim = self.ppo_objs[0].get_shape()
        self.obs_dim, self.act_dim = 4, 8

        # initialize a virtual server
        self.server = method.Actor(self.obs_dim, self.act_dim, kwargs.get("hidden_dim"))

    def aggregate(self):
        # get all the actors in the federation
        actors = [obj.get_actor() for obj in self.ppo_objs]

        # get the state dict of the server
        target_state_dict = self.server.state_dict()

        # average the weights
        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data.fill_(0.)
                for model_id, actor in enumerate(actors):
                    state_dict = actor.state_dict()
                    target_state_dict[key].data += state_dict[key].data.clone() / self.num_clients

        # distribute the averaged weights among the clients
        for id_client in range(self.num_clients):
            actors[id_client].load_state_dict(target_state_dict)

    def train(self):
        
        episodes  = self.kwargs.get("episodes")
        #max_steps = self.kwargs.get("max_steps")
        epochs    = self.kwargs.get("epochs")
        verbose   = self.kwargs.get("verbose")
        freq      = self.kwargs.get("freq")

        rewards_history = np.zeros(episodes)

        # training loop
        for episode in range(episodes): 
            rewards_federation = np.zeros(self.num_clients)   
            for id_client, obj in enumerate(self.ppo_objs):

                # rollout data for the current client
                states, actions, rewards, returns, advantages, log_probs = obj.rollout()

                rewards_federation[id_client] = np.sum(rewards)

            for epoch in range(epochs):
                         mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages = obj.sample_batch(states, actions, log_probs, returns, advantages)

                         obj.update(mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages)

            # the reward of the episode is the average reward of the clients
            rewards_history[episode] = np.mean(rewards_federation)

            
            if verbose >= 1 and episode % 50 == 0:
                print("episode {} --> tot_reward = {}".format(episode, rewards_history[episode]))

            if episode % freq == 0: 
                self.aggregate()
                if episode % 100 == 0:
                    # Save the model and other training-related information
                    checkpoint_path = os.path.join
                    (checkpoint_folder, f'model_checkpoint_episode_{episode}.pth')
                    checkpoint = {
                        'episode': episode,
                        'model_state_dict': self.server.state_dict(),
                    }
                    # Save the checkpoint
                    torch.save(checkpoint, checkpoint_path)

        if verbose >= 2:        
            fig = plt.figure(figsize = (5,5))
            plt.plot(range(episodes), rewards_history)
            plt.xlabel("episode")
            plt.ylabel("reward")
            plt.show()
        
        # close the environments
        for obj in self.ppo_objs: obj.close()

        return rewards_history
    
if __name__ == "__main__":

    # set device to cpu or cuda
    print("=========================================")
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("=========================================") 
     
    parser = argparse.ArgumentParser()
    #parser.add_argument("-e", "--env", type=str, default="CartPole-v1", help="set the gym environment")
    parser.add_argument("-n", "--nclients", type=int, default=4, help="number of clients in the federation")
    parser.add_argument("--episodes", type=int, default=500, help="number of episodes to run")
    parser.add_argument("-s", "--max_steps", type=int, default=100, help="number of steps per episode")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs per episode")
    parser.add_argument("-mb", "--mini_batch_size", type=int, default=6, help="number of samples per mini batch")


    # neural networks parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="number of neurons in the hidden layer")
    parser.add_argument("--step_actor", type=float, default=3e-3, help="step size of the actor optimizer")
    parser.add_argument("--step_critic", type=float, default=3e-3, help="step size of the critic optimizer")

    # variables for the training loop
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="set the discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="set the discount factor")
    parser.add_argument("-l", "--lambda_", type=float, default=1, help="set the lambda value for the GAE")

    # episodic synchronization frequency
    parser.add_argument("-f", "--freq", type=int, default=10, help="episodic syncronization frequency")

    parser.add_argument("-v", "--verbose", action="count", help="show log of rewards", default=0)
    parser.add_argument("--save", type=int, default=0, help="save plot data to a txt file")

    args = parser.parse_args()

    args_dict = vars(args)

    n = args.nclients
    #envs = ["CartPole-v1" for _ in range(n)]
    envs = ["serpentine", "lysegeven"]

    federation = FedPPO(envs, device=device, **args_dict)
    rewards_history = federation.train()

    if args.save:
        if not os.path.exists("./data_ppo"):
        # Create the directory
            os.makedirs("data_ppo")        
        np.savetxt('./data_ppo/n'+str(n)+'.txt', np.column_stack((range(args.episodes), rewards_history)), header='X-axis Y-axis', comments='')
