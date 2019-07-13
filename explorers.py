import math
import numpy as np

class BaseExplorer(object):
    def update(self):
        pass
    
    def choose_action(self, Q):
        pass

class GreedyExplorer(BaseExplorer):
    def choose_action(self, Q):
        return np.argmax(Q)

class EpsilonGreedyExplorer(BaseExplorer):
    def __init__(self, e_initial=1, e_final=0.01, e_decay=0.995):
        self.e = e_initial
        self.e_final = e_final
        self.e_decay = e_decay

    def choose_action(self, Q):
        if np.random.uniform() < self.e:
            return np.random.randint(len(Q))
        else:
            return np.argmax(Q)

    def update(self):
        self.e = max(self.e * self.e_decay, self.e_final)

class BoltzmannExplorer(BaseExplorer):
    def softmax(self, Q):
        exp_sum = sum(math.e**q for q in Q)
        return [math.e**q / exp_sum for q in Q]

    def choose_action(self, vals, normalise=True):
        '''
        `normalise`: whether to apply softmax to Q. For policy gradient
        methods where this is already computed, this can be set to False.
        '''
        action_probs = (self.softmax(vals) if normalise else vals)
        return np.random.choice(np.arange(len(vals)), p=action_probs)
