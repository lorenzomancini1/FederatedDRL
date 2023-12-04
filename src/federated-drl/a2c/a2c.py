import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(Actor, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim), # because we want the actor to return a distribution for the actions
            nn.Softmax(dim = 1) 
        )
        
    def forward(self, state):
        policy = self.actor(state)
        return policy
    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(Critic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        value = self.critic(state)
        return value

def get_env_shape(env):
    try:    act_dim = env.action_space.n
    except: act_dim = env.action_space.shape[0]

    try:    obs_dim = env.observation_space.n
    except: obs_dim = env.observation_space.shape[0]

    return obs_dim, act_dim


def get_returns(rewards, values, mask, gamma, lambda_):
    returns = np.zeros_like(rewards)
    GAE = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * mask[t] - values[t]
        GAE = delta + gamma * lambda_ * GAE * mask[t]
        returns[t] = GAE + values[t]
    return returns

class A2C:
    def __init__(self, env_name, hidden_dim = 256, lr_a = 3e-4, lr_c = 3e-4, gamma = 0.99, lambda_ = 1, device = "cpu"):

        self.device = device
        self.gamma = gamma
        self.lambda_ = lambda_
        #self.max_steps = max_steps
        self.env  = gym.make(env_name)
        self.obs_dim , self.act_dim = get_env_shape(self.env)

        #initialize networks
        self.actor  = Actor(self.obs_dim, self.act_dim, hidden_dim)
        self.critic = Critic(self.obs_dim, hidden_dim)

        #initialize optimizers
        self.optimizer_a = optim.Adam(self.actor.parameters(),  lr = lr_a)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr = lr_c)
    
    def get_actor(self):
        return self.actor
    
    #def update_actor(self, state_dict):
    #    self.actor.load_state_dict(state_dict)
    #
    def get_shape(self):
        return self.obs_dim, self.act_dim

    def close(self):
        self.env.close()

    def rollout(self, max_steps = 100):
        '''
        Collect rollout data by performing a max_steps trajectory
        ''' 
        # initialize empty lists for log_probs, values, rewards and mask
        log_probs = []
        values    = []
        rewards   = []
        mask      = []

        state = self.env.reset()
        #print(state)

        for step in range(1, max_steps + 1):
            policy = self.actor(torch.tensor(state).unsqueeze(0))
            value  = self.critic(torch.tensor(state).unsqueeze(0))

            dist   = policy.detach().numpy()
            # select an action according to the policy
            action   = np.random.choice(self.act_dim, p = np.squeeze(dist))
            # compute the log_prob of that action
            log_prob = torch.log(policy.squeeze(0)[action])

            # do a step
            new_state, reward, done, info = self.env.step(action)

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            mask.append(1 - done)

            state = new_state

            if done: break
        #append the values of the new state for the computations of returns
        values.append(self.critic(torch.tensor(new_state).unsqueeze(0)))
        return rewards, values, log_probs, mask#, new_state


    def update(self, rewards, values, log_probs, mask):
        #new_value = self.critic(torch.tensor(new_state).unsqueeze(0))
        returns   = get_returns(rewards, values, mask, self.gamma, self.lambda_)

        returns   = torch.tensor(returns).to(self.device)
        values    = torch.stack(values).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)

        #compute advantage
        advantage = returns - values

        #compute actor and critic losses
        actor_loss  = - (log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()

        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()
        
        self.optimizer_a.step()
        self.optimizer_c.step()

#training loop
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="CartPole-v1", help="set the gym environment")
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to run")
    parser.add_argument("-s", "--steps", type=int, default=100, help="number of steps per episode")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="set the discount factor")
    parser.add_argument("-l", "--lambda_", type=float, default=1, help="set the lambda value for the GAE")
    parser.add_argument("-v", "--verbose", action="count", help="show log of rewards", default=0)

    args = parser.parse_args()

    env_name = args.env
    print("#################################")
    print("Running:", env_name)
    print("#################################")

    model = A2C(env_name=env_name, gamma=args.gamma, lambda_ = args.lambda_)
    episodes = args.episodes
    max_steps = args.steps

    rewards_history = np.zeros(episodes)
    for episode in range(episodes):
        rewards, values, log_probs, mask = model.rollout(max_steps)

        model.update(rewards, values, log_probs, mask)

        rewards_history[episode] = np.sum(rewards)
        if args.verbose >= 1:
            if episode % 20 == 0:
                print("episode {} --> tot_reward = {}".format(episode, np.sum(rewards)))
    
    model.close()

    if args.verbose >= 2:
        fig = plt.figure(figsize = (5,5))
        plt.plot(range(episodes), rewards_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()
            

            


    

                           

