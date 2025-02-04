from neuron import Neuron

class Layer:

    def __init__(self, d, nn):
        self.ns = [
            Neuron(d) for _ in range(nn)
        ]

    def __call__(self, xs):
        return [
            n(xs) for n in self.ns
        ]

    def ps(self):
        return [p for n in self.ns for p in n.ps()]
