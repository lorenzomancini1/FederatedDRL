import a2c as method
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

#gym.register(
#    id="serpentine",
#    entry_point="veins_gym:VeinsEnv",
#    kwargs={
#        "scenario_dir": "we have to put the correct one",
#    },
#)
#
#
#gym.register(
#    id="lysegeven",
#    entry_point="veins_gym:VeinsEnv",
#    kwargs={
#        "scenario_dir": "we have to put the correct one",
#    },
#)

def get_env_shape(env):
    print(env)
    try:    act_size = env.action_space.n
    except: act_size = env.action_space.shape[0]
    
    try:    obs_size = env.observation_space.n
    except: obs_size = env.observation_space.shape[0]
    
    return obs_size, act_size

class FedA2C:

    def __init__(self, envs, hidden_dim = 256, lr_a = 3e-4, lr_c = 3e-4, gamma = 0.99, lambda_ = 0.95):

        self.gamma   = gamma
        self.lambda_ = lambda_
        self.num_clients = len(envs)

        # initialize A2C objects
        self.a2c_objs = [method.A2C(env_name=env, hidden_dim=hidden_dim, gamma=self.gamma, lambda_=self.lambda_, device=device) for env in envs]

        self.obs_dim, self.act_dim = self.a2c_objs[0].get_shape()

        # initialize a virtual server
        self.server = method.Actor(self.obs_dim, self.act_dim, hidden_dim)
    
    def aggregate(self):
        # get all the actors in the federation
        actors = [obj.get_actor() for obj in self.a2c_objs]

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

    def train(self, episodes = 1000, max_steps = 100, f = 1, v = 2):

        rewards_history = np.zeros(episodes)

        # training loop
        for episode in range(episodes): 
            rewards_federation = np.zeros(self.num_clients)   
            for id_client, obj in enumerate(self.a2c_objs):

                rewards, values, log_probs, mask = obj.rollout(max_steps)
                obj.update(rewards, values, log_probs, mask)

                rewards_federation[id_client] = np.sum(rewards)

            # the reward of the episode is the average reward of the clients
            rewards_history[episode] = np.mean(rewards_federation)

            
            if v >= 1 and episode % 20 == 0:
                print("episode {} --> tot_reward = {}".format(episode, rewards_history[episode]))

            if episode % f == 0: 
                self.aggregate()
                #print(all(torch.equal(param1, param2) for param1, param2 in zip(self.a2c_objs[0].get_actor().state_dict().values(),              self.a2c_objs[1].get_actor().state_dict().values())))

        if v >= 2:        
            fig = plt.figure(figsize = (5,5))
            plt.plot(range(episodes), rewards_history)
            plt.xlabel("episode")
            plt.ylabel("reward")
            plt.show()
        return rewards_history

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to run")
    parser.add_argument("-s", "--steps", type=int, default=100, help="number of steps per episode")
    parser.add_argument("-n", "--nclients", type=int, default=4, help="number of clients in the federation")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="set the discount factor")
    parser.add_argument("-l", "--lambda_", type=float, default=1, help="set the lambda value for the GAE")
    parser.add_argument("-f", "--freq", type=int, default=10, help="episodic syncronization frequency")
    parser.add_argument("-v", "--verbose", action="count", help="show log of rewards", default=0)
    parser.add_argument("--save", type=int, default=0, help="save plot data to a txt file")

    args = parser.parse_args()

   #envs = ["serpentine", "lysegeven"] #, "CartPole-v1"]
    n = args.nclients
    envs = ["CartPole-v1" for _ in range(n)]
    episodes = args.episodes
    freq = args.freq
    max_steps = args.steps
    verbose = args.verbose

    federation = FedA2C(envs, gamma=args.gamma, lambda_=args.lambda_)
    rewards_history = federation.train(episodes, max_steps, freq, verbose)

    if args.save:
        if not os.path.exists("./data_a2c"):
        # Create the directory
            os.makedirs("data_a2c")        
        np.savetxt('./data_a2c/data.txt', np.column_stack((range(episodes), rewards_history)), header='X-axis Y-axis', comments='')

    


