import numpy as np
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
def reinforceWithBaseline(theta,w,mdp,state):
    converged=False
    s=state[np.newaxis]
    v=np.dot(np.array(w).transpose(),state)
    pi=softmax(np.dot(theta,state))
    print("vi:")
    print(v)
    print("policy:")
    print(pi)
    return None