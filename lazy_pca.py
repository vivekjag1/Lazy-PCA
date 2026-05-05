import numpy as np
from numpy.linalg import norm
from math import sin, cos, pi

from power_iteration import PowerIteration


class LazyPCA():

    def __init__(self, theta_thresh : float, num_iter = 200, epsilon = 1e-6):
        '''
        Constructs LazyPCA
        theta_thresh - angle for near-orthogonality flexibility measured in degrees
        '''
        self.theta_thresh = theta_thresh
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.iter = PowerIteration()
        

    def _cosine_sim(self, x1, x2) -> float:
        '''
        calculates cosine similarity between vectors
        '''
        return np.dot(x1,x2) / (norm(x1)*norm(x2))
    
    def _remove_var(self, X, w):
        '''
        computes X with most of the variance along w removed
        '''
        return X - cos(pi*self.theta_thresh/180) * np.outer(X @ w, w)

    def find_top_k(self, X, k : int):
        '''
        find the top k components
        Uses lazy calculation where not all of the variance is removed
        '''
        components = []
        for i in range(k):
            components.append(self.iter.power_iteration_single_iteration(X))

            w = components[i]
            X = self._remove_var(X,w)
        
        return components
    
    def find_top_k_strict(self, X, k : int):
        '''
        find the top k components with stricter adherence to the theta_threshold
        Simple incomplete variance elimination can result in not enough variance being removed,
            so this repeats the variance reduction until it's far enough away
        '''
        components = []
        for i in range(k):
            new_w = self.iter.power_iteration_single_iteration(X)
            for old_w in components:
                while self._cosine_sim(new_w, old_w) >= sin(pi*self.theta_thresh/180):
                    # need to scale down the variance in old_w's direction more to prevent angles being too close
                    X = self._remove_var(X, old_w)
                    # then recalculate the new vector
                    new_w = self.iter.power_iteration_single_iteration(X)
            # now it's guaranteed that the new vector is near-orthogonal to the old vectors
            components.append(new_w)
            X = self._remove_var(X, new_w)
        
        return components
    
    def fit_transform(self, X, k : int, use_strict=True):
        '''
        finds the top k components and returns (xform_data, comps) as a tuple
        '''

        find_k_func = self.find_top_k_strict if use_strict else self.find_top_k

        comps = np.array(find_k_func(X, k))
        xform_data = X.dot(comps.T)
        return (xform_data, comps)
