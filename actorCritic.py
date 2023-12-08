import torch
import torch.nn as nn
import torch.optim as optim
import gym

from plotGraph import plot_graph
from network import Actor,Critic

class ActorCriticAgent:
    def __init__(self, env, actor, critic, actor_optimizer, critic_optimizer, gamma=1):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)

        # Compute TD Error
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else torch.tensor([0.0], dtype=torch.float32)
        td_error = reward + self.gamma * next_value - value

        # Update Critic
        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor using Policy Gradient
        action_probs = self.actor(state)
        selected_action_prob = action_probs.gather(1, action.unsqueeze(1))
        actor_loss = -(torch.log(selected_action_prob) * td_error.detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# Training loop
def train_actor_critic(agent, params):
    rewards = []
    num_episodes=params["num_episodes"]

    for episode in range(num_episodes):
        state = agent.env.reset()[0]
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _,_ = agent.env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        rewards.append(total_reward)
    return rewards

# Instantiate the environment and the Actor-Critic agent
def actor_critic(params):
    # Hyperparameters
    env_name=gym.make(params["mdp"])
    input_size = env_name.observation_space.shape[0]
    output_size = env_name.action_space.n
    hidden_size = [128]
    learning_rate_actor = 0.001
    learning_rate_critic = 0.01
    # Instantiate networks and optimizer
    actor = Actor(input_size, hidden_size, output_size,1)
    critic = Critic(input_size, hidden_size,1,1)
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate_critic)

    agent = ActorCriticAgent(env_name, actor, critic, actor_optimizer, critic_optimizer)

    rewards = train_actor_critic(agent,params)


    plot_graph(rewards,params,'Episode','Total Reward')
