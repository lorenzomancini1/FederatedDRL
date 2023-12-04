
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse
#from tabulate import tabulate

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
    '''
    Compute the returns using the GAE formula
    '''
    returns = np.zeros_like(rewards)
    GAE = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * mask[t] - values[t]
        GAE = delta + gamma * lambda_ * GAE * mask[t]
        returns[t] = GAE + values[t]
    return returns

class PPO:
    def __init__(self, env, device, **kwargs):

        # hyperparameters
        self.gamma   = kwargs.get("gamma")
        self.lambda_ = kwargs.get("lambda_")
        self.epsilon = kwargs.get("epsilon")

        self.max_steps = kwargs.get("max_steps")

        self.mini_batch_size = kwargs.get("mini_batch_size", 6)
        hidden_dim           = kwargs.get("hidden_dim", 256)
        lr_a                 = kwargs.get("step_actor")
        lr_c                 = kwargs.get("step_critic")

        # set the proper device
        self.device = device

        # make the gym environment and get the observation and action dimensions
        #self.env    = gym.make(kwargs.get("env_name"))
        self.env = gym.make(env)
        #self.obs_dim , self.act_dim = get_env_shape(self.env)
        self.obs_dim, self.act_dim = 4, 8
        
        #initialize networks
        self.actor  = Actor(self.obs_dim, self.act_dim, hidden_dim)
        self.critic = Critic(self.obs_dim, hidden_dim)

        #initialize optimizers
        self.optimizer_a = optim.Adam(self.actor.parameters(),  lr = lr_a)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr = lr_c)

    def get_actor(self):
        return self.actor
    
    def close(self):
        self.env.close()

    def get_shape(self):
        return self.obs_dim, self.act_dim

    def rollout(self):
        '''
        Collect data by performing a n-step trajectory
        '''

        env = self.env
        max_steps = self.max_steps

        states    = np.empty(max_steps, dtype = object)
        actions   = np.zeros(max_steps, dtype = int)
        rewards   = np.zeros(max_steps, dtype = float)
        values    = np.zeros(max_steps + 1, dtype = float)
        log_probs = np.zeros(max_steps, dtype = float)
        mask      = np.zeros(max_steps, dtype = int)

        state = env.reset()

        for step in range(max_steps):

            # we don't need the gradients for the moment. We will compute them again in the epochs loop
            with torch.no_grad():
                policy = self.actor(torch.tensor(state).unsqueeze(0))
                value  = self.critic(torch.tensor(state).unsqueeze(0))

            dist = policy.numpy() # set the policy to numpy vector

            # select an action according to the policy
            action   = np.random.choice(self.act_dim, p = np.squeeze(dist))
            # compute the log_prob of that action
            log_prob = torch.log(policy.squeeze(0)[action])

            # do a step
            new_state, reward, done, info = self.env.step(action)

            states[step]    = state
            actions[step]   = action
            rewards[step]   = reward
            values[step]    = value
            log_probs[step] = log_prob
            mask[step]      = 1 - done
        
            state = new_state
            if done: break
        
        stop = step + 1
        
        # to compute the returns we need the value of the (n+1)-th state
        with torch.no_grad():
            values[stop] = self.critic(torch.tensor(new_state).unsqueeze(0))
            
        states    = states[:stop]
        actions   = actions[:stop]
        rewards   = rewards[:stop]
        values    = values[:stop + 1] # we save the new state too for the computation of the returns
        log_probs = log_probs[:stop]
        mask      = mask[:stop]

        returns = get_returns(rewards, values, mask, self.gamma, self.lambda_)

        # computation of the advantages 
        advantages = returns - values[:-1]

        return states, actions, rewards, returns, advantages, log_probs
    
    def update(self, states, actions, log_probs, returns, advantages):
        '''
        Loop over the elements of a minibatch and update the Actor and the Critic for k-epochs
        '''

        mini_batch_size = self.mini_batch_size
    
        actor_loss  = torch.empty(mini_batch_size)#, dtype = object)
        critic_loss = torch.empty(mini_batch_size)

        for i in range(mini_batch_size):
            # get the current state and action
            state  = states[i]
            action = actions[i]

            # compute the policy and the value
            policy = self.actor(torch.tensor(state).unsqueeze(0).to(self.device))
            value  = self.critic(torch.tensor(state).unsqueeze(0).to(self.device))

            # select the current advantage and return
            adv     = advantages[i]
            return_ = returns[i]

            # compute the current log probability and the old one
            curr_log_prob = torch.log(policy.squeeze(0)[action])
            old_log_prob  = log_probs[i]

            # compute the ratio beween new and old log prob
            ratio = (curr_log_prob - old_log_prob).exp()
            s1    = ratio * adv
            s2    = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

            actor_loss[i]  = torch.min(s1, s2) 
            critic_loss[i] = return_ - value

        # compute the losses
        epoch_actor_loss  = - actor_loss.mean()
        epoch_critic_loss = critic_loss.pow(2).mean()

        # update the networks
        self.optimizer_a.zero_grad()
        self.optimizer_c.zero_grad()

        epoch_actor_loss.backward()
        epoch_critic_loss.backward()

        self.optimizer_a.step()
        self.optimizer_c.step()
    
    def sample_batch(self, states, actions, log_probs, returns, advantages):
        '''
        Sample a minibatch from the trajectory
        '''

        assert len(states) == len(log_probs) == len(returns) == len(advantages)

        high = len(states)
        n = self.mini_batch_size

        # get some random indexes for the trajectory
        random_idxs = np.random.randint(0, high, n)

        mb_states     = states[random_idxs]
        mb_actions    = actions[random_idxs]
        mb_log_probs  = log_probs[random_idxs]
        mb_returns    = returns[random_idxs]
        mb_advantages = advantages[random_idxs]

        return mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages
    
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

    # environment
    parser.add_argument("-e", "--env_name", type=str, default="CartPole-v1", help="set the gym environment")

    # variables for the training loop
    parser.add_argument("--episodes", type=int, default=1000, help="number of episodes to run")
    parser.add_argument("-s", "--max_steps", type=int, default=100, help="number of steps per episode")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs per episode")
    parser.add_argument("-mb", "--mini_batch_size", type=int, default=6, help="number of samples per mini batch")

    # neural networks parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="number of neurons in the hidden layer")
    parser.add_argument("-sa", "--step_actor", type=float, default=3e-3, help="step size of the actor optimizer")
    parser.add_argument("-sc", "--step_critic", type=float, default=3e-3, help="step size of the critic optimizer")

    # policy gradient variables
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="set the discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="set the discount factor")
    parser.add_argument("-l", "--lambda_", type=float, default=0.95, help="set the lambda value for the GAE")
    
    # log of rewards and plot 
    parser.add_argument("-v", "--verbose", action="count", help="show log of rewards", default=0)

    args = parser.parse_args()

    args_dict = vars(args)

    env_name = args.env_name
    
    #print("#################################")
    #print("Running:", env_name)
    #print("#################################")

    # Convert the dictionary directly to a list of tuples
    #table_data = list(args_dict.items())

    ## Print the table using tabulate
    #print(tabulate(table_data, headers=["Key", "Value"], tablefmt="fancy_grid"))

    model = PPO(env = env_name, device = device, **args_dict)
    episodes = args.episodes

    ###########################
    ###### TRAINING LOOP ######
    ###########################
    rewards_history = np.zeros(episodes)
    for episode in range(episodes):

        states, actions, rewards, returns, advantages, log_probs = model.rollout()

        rewards_history[episode] = np.sum(rewards)
        if args.verbose >= 1:
            if episode % 50 == 0:
                print("episode {} --> tot_reward = {}".format(episode, np.sum(rewards)))

        for epoch in range(args.epochs):
             mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages = model.sample_batch(states, actions, log_probs, returns, advantages)

             model.update(mb_states, mb_actions, mb_log_probs, mb_returns, mb_advantages)

    model.close()

    if args.verbose >= 2:
        fig = plt.figure(figsize = (5,5))
        plt.plot(range(episodes), rewards_history)
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.show()

