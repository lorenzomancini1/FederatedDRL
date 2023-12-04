import ppo as method
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
import gym

import argparse
import os

# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class FedPPO:

    def __init__(self, envs, lr_a = 3e-3, lr_c = 3e-3, device = device, **kwargs):
        
        self.num_clients = len(envs)

        # initialize PPO objects
        self.ppo_objs = [method.PPO(env_name=env, **kwargs) for env in envs]

        self.obs_dim, self.act_dim = self.ppo_objs[0].get_shape()

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

    def train(self, episodes = 1000, max_steps = 100, epochs = 4, f = 10, v = 2):
        
        rewards_history = np.zeros(episodes)

        # training loop
        for episode in range(episodes): 
            rewards_federation = np.zeros(self.num_clients)   
            for id_client, obj in enumerate(self.ppo_objs):

                # rollout data for the current client
                states, actions, rewards, returns, advantages, log_probs = obj.rollout(max_steps = 150)

                rewards_federation[id_client] = np.sum(rewards)

            for epoch in range(epochs):
                         mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages = obj.sample_batch(states, actions, log_probs, returns, advantages)

                         obj.update(mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages)

            # the reward of the episode is the average reward of the clients
            rewards_history[episode] = np.mean(rewards_federation)

            
            if v >= 1 and episode % 20 == 0:
                print("episode {} --> tot_reward = {}".format(episode, rewards_history[episode]))

            if episode % f == 0: 
                self.aggregate()

        if v >= 2:        
            fig = plt.figure(figsize = (5,5))
            plt.plot(range(episodes), rewards_history)
            plt.xlabel("episode")
            plt.ylabel("reward")
            plt.show()
        return rewards_history
    
if __name__ == "__main__":
     
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="CartPole-v1", help="set the gym environment")
    parser.add_argument("-n", "--nclients", type=int, default=4, help="number of clients in the federation")
    parser.add_argument("--episodes", type=int, default=500, help="number of episodes to run")
    parser.add_argument("--hidden_dim", type=int, default=256, help="number of neurons in the hidden layer")
    parser.add_argument("-s", "--steps", type=int, default=100, help="number of steps per episode")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="set the discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="set the discount factor")
    parser.add_argument("-l", "--lambda_", type=float, default=1, help="set the lambda value for the GAE")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs per episode")
    parser.add_argument("-f", "--freq", type=int, default=10, help="episodic syncronization frequency")
    parser.add_argument("-v", "--verbose", action="count", help="show log of rewards", default=0)
    parser.add_argument("--save", type=int, default=0, help="save plot data to a txt file")

    args = parser.parse_args()

    args_dict = vars(args)

    n = args.nclients
    envs = ["CartPole-v1" for _ in range(n)]
    episodes = args.episodes
    freq = args.freq
    epochs = args.epochs
    max_steps = args.steps
    verbose = args.verbose

    federation = FedPPO(envs, device=device, **args_dict)
    rewards_history = federation.train(episodes, max_steps, epochs, freq, verbose)


#self, episodes = 1000, max_steps = 100, epochs = 4, f = 10, v = 2