import numpy as np
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
def reinforceWithBaseline(theta,w,mdp,state):
    converged=False
    s=state.reshape(4,1)
    v=np.dot(w.reshape((9, 1)),s.transpose())
    pi=softmax(np.dot(theta,s))
    print("vi:")
    print(v)
    print("policy:")
    print(pi)
    return None