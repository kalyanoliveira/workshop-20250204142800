from value import Value

import numpy as np
np.random.seed(42)

class Neuron:

    def __init__(self, d):
        self.ws = [
            Value(np.random.uniform(-1, 1)) for _ in range(d)
        ]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, xs):
        return sum(
            [
                x*w for x, w in zip(xs, self.ws)
            ] 
            + [self.b]
        ).tanh()

    def ps(self):
        return self.ws + [self.b]
