import gym
from network import Policy,ValueFunction
import torch.optim as optim
import torch.nn as nn
import torch

from plotGraph import plot_graph
import matplotlib.pyplot as plt

def reinforce_baseline(params,gamma=1,alpha=0.001,alpha_w=0.001):
    env_name=params["mdp"]
    num_episodes=params["num_episodes"] 
    env = gym.make(env_name)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    policy = Policy(input_size, [128,128], output_size,3)
    value_function = ValueFunction(input_size, [128,128],1,3)

    policy_optimizer = optim.Adam(policy.parameters(), lr=alpha)
    value_optimizer = optim.Adam(value_function.parameters(), lr=alpha_w)

    episode_cumulative_rewards = []
    actions_taken=[]
    episode_values=[]
    actions=0
    reward_plot=0
    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []
        values = []
        v=[]

        while True:
            actions+=1
            state = torch.FloatTensor(state).unsqueeze(0)
            action_prob = policy(state)
            value = value_function(state)

            action = torch.multinomial(action_prob, 1).item()
            log_prob = torch.log(action_prob[0, action])

            state, reward, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            v.append(value.item())

            if done:
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
        reward_plot+=sum(rewards)
        episode_cumulative_rewards.append(reward_plot)
        actions_taken.append(actions)
        episode_values.append(v)
    
    env.close()
    # print(episode_rewards)
    plot_graph(episode_cumulative_rewards,params,'Episode','Cumulative Reward')
    #plot_graph(actions_taken,params,'Episodes','Actions Taken')
    # plt.figure(figsize=(10, 5))
    # for i, values in enumerate(episode_values):
    #     plt.plot(range(len(v)), v, label=f'Episode {i + 1}')
    # plt.xlabel('Time Step')
    # plt.ylabel('Estimated Value')
    # plt.title('Value Estimates Across Trajectory for Each Episode')
    # plt.legend()
    # plt.show()