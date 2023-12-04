import numpy as np
def normalize(inp,range_init:tuple[float, float],range_fin:tuple[float, float]):
    res = (inp - range_init[0]) / (range_init[1] - range_init[0])
    res = res * (range_fin[1] - range_fin[0]) + range_fin[0]
    return res

def normalize_state(state:np.ndarray, mdp_param, mdp, range):
    if mdp==1:
        print(mdp_param)
        return np.array([
                normalize(state[0], (mdp_param["X_MIN"], mdp_param["X_MAX"]), range),
                normalize(state[1], mdp_param["V_RANGE"], range),
                normalize(state[2], (mdp_param["OMEGA_MIN"], mdp_param["OMEGA_MAX"]), range),
                normalize(state[3], mdp_param["OMEGA_DOT_RANGE"], range)
            ])
    else:
        return np.array([
                normalize(state[0], (mdp_param["X_MIN"], mdp_param["X_MAX"]), range),
                normalize(state[1], mdp_param["V_RANGE"], range),
            ])

def sine_basis(state:np.ndarray, mdp_param, mdp,order):
    angles = np.array([i*np.pi for i in range(1, order+1)]).reshape(1, -1)
    feature = normalize_state(state, mdp_param, mdp,(-1.0, 1.0)).reshape(-1, 1)
    basis = np.matmul(feature, angles).flatten()
    basis = np.sin(basis)
    basis = np.insert(basis, 0, 1.0)
    return basis

def cosine_basis(state:np.ndarray, mdp_param, mdp,order):
    angles = np.array([i*np.pi for i in range(order+1)]).reshape(1, -1)
    feature = normalize_state(state, mdp_param, mdp, (0.0, 1.0)).reshape(-1, 1)
    basis = np.matmul(feature, angles).flatten()
    basis = np.cos(basis)
    basis = np.insert(basis, 0, 1.0)
    return basis