from collections import deque
import numpy as np

class BaseExperience(object):
    def __init__(self, max_experiences=100000):
        self.max_experiences = max_experiences

    def add(self, s, a, r, s_, done):
        pass
 
    def sample(self, n):
        pass

class PrioritisedExperience(BaseExperience):
    # TODO: implement as sum-tree
    def __init__(self, alpha=0.09, epsilon=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = {}
        self.idx = 0

    def add(self, s, a, r, s_, done, error):
        self.buffer.update({self.idx: [s, a, r, s_, done, error]})
        if self.idx >= self.max_experiences:
            del self.buffer[self.idx-self.max_experiences]
        self.idx += 1

    def set_error(self, idx, error):
        self.buffer[idx][-1] = error
 
    def sample(self, n):
        if len(self.buffer) <= n:
            return {}

        probs = np.array([self.priority(error)**self.alpha for _, (*_, error)
                          in self.buffer.items()])
        probs /= sum(probs)
        idxs = np.random.choice(np.arange(len(self.buffer)), size=n, p=probs)
        samples = [(idx, *self.buffer[idx]) for idx in idxs]
        return samples

    def priority(self, error):
        return abs(error) + self.epsilon
