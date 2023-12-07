import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class ValueFunction(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size=1):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)

def reinforce_baseline(env_name, num_episodes, gamma=0.99, learning_rate_policy=0.01, learning_rate_value=0.01, hidden_size_policy1=64, hidden_size_policy2=32, hidden_size_value1=64, hidden_size_value2=32):
    env = gym.make(env_name)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # Instantiate the more complex Policy and ValueFunction
    policy = Policy(input_size, hidden_size_policy1, hidden_size_policy2, output_size)
    value_function = ValueFunction(input_size, hidden_size_value1, hidden_size_value2)

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate_policy)
    value_optimizer = optim.Adam(value_function.parameters(), lr=learning_rate_value)

    # ... (rest of the function remains unchanged)
    episode_rewards = []  # Store rewards for each episode

    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []
        values = []

        while True:
            state = torch.FloatTensor(state).unsqueeze(0)
            action_prob = policy(state)
            value = value_function(state)

            action = torch.multinomial(action_prob, 1).item()
            log_prob = torch.log(action_prob[0, action])

            state, reward, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

            if done:
                break

        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        advantages = returns - torch.cat(values, dim=1).detach()

        # Policy gradient update
        policy_loss = -torch.sum(torch.stack(log_probs) * advantages.squeeze(1))
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Value function update
        value_loss = nn.MSELoss()(torch.cat(values), returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        episode_rewards.append(sum(rewards))

        # if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

    env.close()
    return episode_rewards

# Run REINFORCE with baseline on CartPole
#cartpole_rewards = reinforce_baseline('CartPole-v1', num_episodes=3000)

# Run REINFORCE with baseline on MountainCar
mountaincar_rewards = reinforce_baseline('MountainCar-v0', num_episodes=500)

# Plotting
plt.figure(figsize=(12, 6))
#plt.plot(cartpole_rewards, label='CartPole')
plt.plot(mountaincar_rewards, label='MountainCar')
plt.title('REINFORCE with Baseline Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.show()

