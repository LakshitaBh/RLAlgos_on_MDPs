import gym
from gym.core import RewardWrapper
from network import Policy,ValueFunction
import torch.optim as optim
import torch.nn as nn
import torch

from plotGraph import plot_graph
import matplotlib.pyplot as plt

def reinforce_baseline(params,gamma=1,alpha=0.001,alpha_w=0.001):
    seed = 0
    env_name=params["mdp"]
    num_episodes=params["num_episodes"] 
    env = gym.make(env_name)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    policy = Policy(input_size, [64,64], output_size,2)
    value_function = ValueFunction(input_size, [64,64],1,2)
    torch.manual_seed(seed)

    policy_optimizer = optim.Adam(policy.parameters(), lr=alpha)
    value_optimizer = optim.Adam(value_function.parameters(), lr=alpha_w)
    rewards_array=[]
    cumulative_reward=0
    cumulative_rewards=[]
    average_reward=[]
    no_steps=[]
    cumulative_steps=[]
    total_steps=0
    for episode in range(num_episodes):
        print("episode=",episode)
        state = env.reset(seed=seed)[0]
        log_probs = []
        rewards = []
        values = []
        episode_step=0
        while True:
            episode_step+=1
            state = torch.FloatTensor(state).unsqueeze(0)
            action_prob = policy(state)
            value = value_function(state)
            action = torch.multinomial(action_prob, 1).item()
            log_prob = torch.log(action_prob[0, action])
            state, reward, done, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            
            if done or truncated:
                break

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        delta = returns - torch.cat(values, dim=1).detach()


        w= nn.MSELoss()(torch.cat(values), returns)
        value_optimizer.zero_grad()
        w.backward()
        value_optimizer.step()

        theta= -torch.sum(torch.stack(log_probs) * delta.squeeze(1))
        policy_optimizer.zero_grad()
        theta.backward()
        policy_optimizer.step()


        rewards_array.append(G)
        cumulative_reward+=G
        cumulative_rewards.append(cumulative_reward)
        average_reward.append(cumulative_reward/(episode+1))
        total_steps+=episode_step
        no_steps.append(episode_step)
        cumulative_steps.append(total_steps)

    env.close()  
    return rewards_array, cumulative_rewards, average_reward, no_steps, cumulative_steps
    #return no_steps