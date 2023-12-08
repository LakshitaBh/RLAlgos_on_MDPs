from reinforceWithBaseline import reinforce_baseline
from actorCritic import actor_critic
import sys

def call_algo(params):
    if params["algo"]==1:
        params["title"]='REINFORCE with Baseline Training Progress'
        reinforce_baseline(params)
    else:
        params["title"]='Actor Critic Training Progress'
        actor_critic(params)

def main(inputs=[2,1,500]):
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


