import numpy as np # version 2.4.2

# constants 
NUM_ITERATIONS = 2000
EPSILON = 1e-6 

# this will need to be tuned 
NUM_COMPONENTS = 3

# power iteration to find a single component. 
def power_iteration(X):
    # find M (this is just XTX)
    M = X.T @ X

    # find random vector V in the shpe of M 
    rng = np.random.default_rng()
    v = rng.standard_normal(M.shape[0])
    # make v a unit vector 
    v0 = v / np.linalg.norm(v)

    v_prev = None

    for i in range(NUM_ITERATIONS): 
        # if i = 0, skip squaring 
        if i>0: 
            M = M @ M 

        # find new V 
        v_new = M @ v0

        # normalize V 
        v_new = v_new / np.linalg.norm(v_new)

        # convergence check 
        if v_prev is not None and np.linalg.norm(v_new - v_prev) < EPSILON:
            return v_new 
        v_prev = v_new  
    return v_prev




if __name__ == "__main__": 
    # at some point, we'll have embeddings. right now we dont, so im making a dummy X matrix 
    X = np.random.rand(5,2)

    components = []

    for i in range(NUM_COMPONENTS): 
        # find each component
        components.append(power_iteration(X))

        w = components[i]
        # computer the projection of X onto W, then subtract it from X
        X = X - np.outer(X @ w, w)
    print(components)

    





