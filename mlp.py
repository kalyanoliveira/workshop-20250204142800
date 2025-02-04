from layer import Layer

class MLP:
    def __init__(self, ni, nns):
        self.ls = [
            Layer(d, nn) for d, nn
                in zip(
                    ([ni] + nns)[:-1],
                    nns
                )
        ]

    def __call__(self, xs):
        x = xs
        for l in self.ls:
            x = l(x)
        return x

    def ps(self):
        return [p for l in self.ls for p in l.ps()]
