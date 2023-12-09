from reinforceWithBaseline import reinforce_baseline
from actorCritic import actor_critic
from plotGraph import plot_graph

def call_algo(params):
    if params["algo"]==1:
        params["title"]='REINFORCE with Baseline Training Progress'
        # average_reward = reinforce_baseline(params,1,0.001,0.001)
        # plot_graph([average_reward]," Average Rewards for"+params["label"],[0.001],"Episodes","Rewards")
        # #plot_graph(no_steps_list,"Number of Steps for"+params["label"]+"for different alpha_theta",[0.001],"Episodes","Steps")
        alpha_theta_values=[0.001,0.0001,0.00001]
        alpha_w_values=[0.001,0.0001,0.00001]
        rewards_array_list, cumulative_rewards_list, average_reward_list, no_steps_list, cumulative_steps_list=[],[],[],[],[]
        for i in alpha_theta_values:
            rewards_array, cumulative_rewards, average_reward, no_steps, cumulative_steps = reinforce_baseline(params,1,i,0.001)
            rewards_array_list.append(rewards_array)
            cumulative_rewards_list.append(cumulative_rewards)
            average_reward_list.append(average_reward)
            no_steps_list.append(no_steps)
            cumulative_steps_list.append(cumulative_steps)
        plot_graph(rewards_array_list,"Rewards for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Rewards")
        plot_graph(cumulative_rewards_list,"Cumulative Rewards for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Cumulative Rewards")
        plot_graph(average_reward_list,"Average Rewards for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Average Rewards")
        plot_graph(no_steps_list,"Number of Steps for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Steps")
        plot_graph(cumulative_steps_list,"Cumulative Number of Steps for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Cumulative Steps")
        for i in alpha_w_values:
            rewards_array, cumulative_rewards, average_reward, no_steps, cumulative_steps = reinforce_baseline(params,1,0.001,i)
            rewards_array_list.append(rewards_array)
            cumulative_rewards_list.append(cumulative_rewards)
            average_reward_list.append(average_reward)
            no_steps_list.append(no_steps)
            cumulative_steps_list.append(cumulative_steps)
        plot_graph(rewards_array_list,"Rewards for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Rewards")
        plot_graph(cumulative_rewards_list,"Cumulative Rewards for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Cumulative Rewards")
        plot_graph(average_reward_list,"Average Rewards for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Average Rewards")
        plot_graph(no_steps_list,"Number of Steps for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Steps")
        plot_graph(cumulative_steps_list,"Cumulative Number of Steps for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Cumulative Steps")
    else:
        params["title"]='Actor Critic Training Progress'
        # average_reward_list = actor_critic(params,1,0.001,0.001)
        # plot_graph(average_reward_list,"Average Rewards for"+params["label"],[0.001],"Episodes","Average Rewards")
        # #plot_graph(rewards_array_list,"Rewards for"+params["label"],[0.001],"Episodes","Rewards")
        alpha_theta_values=[0.001,0.0001,0.00001]
        alpha_w_values=[0.001,0.0001,0.00001]
        rewards_array_list, cumulative_rewards_list, average_reward_list, no_steps_list, cumulative_steps_list=[],[],[],[],[]
        for i in alpha_theta_values:
            rewards_array, cumulative_rewards, average_reward, no_steps, cumulative_steps = actor_critic(params,0.9,i,0.001)
            rewards_array_list.append(rewards_array)
            cumulative_rewards_list.append(cumulative_rewards)
            average_reward_list.append(average_reward)
            no_steps_list.append(no_steps)
            cumulative_steps_list.append(cumulative_steps)
        plot_graph(rewards_array_list,"Rewards for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Rewards")
        plot_graph(cumulative_rewards_list,"Cumulative Rewards for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Cumulative Rewards")
        plot_graph(average_reward_list,"Average Rewards for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Average Rewards")
        plot_graph(no_steps_list,"Number of Steps for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Steps")
        plot_graph(cumulative_steps_list,"Cumulative Number of Steps for"+params["label"]+"for different alpha_theta",alpha_theta_values,"Episodes","Cumulative Steps")
        for i in alpha_w_values:
            rewards_array, cumulative_rewards, average_reward, no_steps, cumulative_steps = actor_critic(params,0.9,0.001,i)
            rewards_array_list.append(rewards_array)
            cumulative_rewards_list.append(cumulative_rewards)
            average_reward_list.append(average_reward)
            no_steps_list.append(no_steps)
            cumulative_steps_list.append(cumulative_steps)
        plot_graph(rewards_array_list,"Rewards for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Rewards")
        plot_graph(cumulative_rewards_list,"Cumulative Rewards for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Cumulative Rewards")
        plot_graph(average_reward_list,"Average Rewards for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Average Rewards")
        plot_graph(no_steps_list,"Number of Steps for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Steps")
        plot_graph(cumulative_steps_list,"Cumulative Number of Steps for"+params["label"]+"for different alpha_w",alpha_w_values,"Episodes","Cumulative Steps")

def main(inputs=[1,2,500]):
    if inputs[0]==1:
        params={
            "mdp":"CartPole-v1",
            "algo":inputs[1],
            "num_episodes":inputs[2],
            "label":'CartPole'

        }
    else:
        params={
            "mdp":"Acrobot-v1",
            "algo":inputs[1],
            "num_episodes":inputs[2],
            "label":'Acrobot'
        }
    call_algo(params)

if __name__=='__main__':
    main()


