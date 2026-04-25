import numpy as np # version 2.4.2
import pandas as pd 
# constants 
NUM_ITERATIONS = 200
EPSILON = 1e-6 
# this will need to be tuned 
NUM_COMPONENTS = 50



class PowerIteration(): 
    def __init__(self, num_iterations, epsilon, num_components, features): 
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.num_components = num_components
        self.features = features



    # power iteration to find a single component. 
    def power_iteration_single_iteration(self):

        # find M (this is just XTX). Instead of repeatedly finding M, determine it once at the top of the function 
        M = self.features.T @ self.features

        # find random vector V in the shpe of M 
        rng = np.random.default_rng()

        # find a random vector
        v = rng.standard_normal(M.shape[0])

        # make v a unit vector 
        v = v / np.linalg.norm(v)

        # hold the previous vector 
        v_prev = None

        # iteration to find more and more vectors 
        for i in range(self.num_iterations): 

            # find the new vector
            v_new = M @ v 

            # normalize the vector to make it a unit vector 
            v_new = v_new / np.linalg.norm(v_new)

            # convergence check 
            if v_prev is not None and np.linalg.norm(v_new - v_prev) < self.epsilon:
                return v_new

            # update the previous vector 
            v_prev = v_new  

            # update the new dimension 
            v = v_new
        return v_prev

    # iterative function 
    def power_iteration(self): 

        # component list 
        components = []

        # find each component
        for i in range(self.num_components): 

            # find the next component
            components.append(self.power_iteration_single_iteration())
            w = components[i]

            # remove the component we just found from the data

            self.features = self.features - np.outer(self.features @ w, w)
        return components


if __name__ == "__main__": 

    # read into a cata frame
    df = pd.read_csv('med_data_with_embeds.csv')

    # drop everything except the embeddings
    embeddings = np.vstack(
        df['embeds'].apply(lambda s: np.fromstring(s.strip('[]'), sep=' ')).values
    )

    # center the data 
    embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)

    # debug output
    # print(f"Embeddings shape: {embeddings.shape}")

    # make the power iteration 
    power_iter = PowerIteration(NUM_ITERATIONS, EPSILON, NUM_COMPONENTS, embeddings)

    # find the components
    compontents = power_iter.power_iteration()

    print(compontents)


