import gym
from fourier import sine_basis
from reinforce import reinforceWithBaseline
import numpy as np
def main():
    mountaincar=gym.make('MountainCar-v0')
    cartpole=gym.make('CartPole-v1')
    state=np.array([0,0,0,0])
    mdp_range={
        'X_MAX':2.4,
        "X_MIN":-2.4,
        'V_RANGE':[-np.pi/15,np.pi/15],
        "OMEGA_MIN":-2*np.pi/15,
        "OMEGA_MAX":2*np.pi/15,
        "OMEGA_DOT_RANGE":[-np.pi/2,np.pi/2]
        }
    w=sine_basis(state,mdp_range,1,2)
    theta=np.random.normal(0.0,0.1,(2,len(w)))
    reinforceWithBaseline(theta,w,mountaincar,state)

if __name__=='__main__':
    main()