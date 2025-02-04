import numpy as np

class Value:

    def __init__(
        self,
        data,
        prev = (),
        op = "",
        label = "",
    ):
        self.data = data
        self.prev = set(prev)
        self.op = op
        self.label = label
        self.grad = 0
        self.back = lambda: None 

    def backward(self):
        topo = []
        visited = set()
        def build_topo(n):
            if n not in visited:
                visited.add(n)
                for p in n.prev:
                    build_topo(p)
            topo.append(n)
        build_topo(self)
        self.grad = 1
        for n in reversed(topo):
            n.back()

    def __repr__(self):
        return f"Value({self.data=})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )
        def b():
            self.grad += out.grad
            other.grad += out.grad
        out.back = b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )
        def b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.back = b
        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self, ), "tanh")
        def b():
            self.grad += out.grad * ( 1 - (out.data)**2 )
        out.back = b
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(
            self.data**other, 
            (self, ), 
            f"**{other}"
        )
        def b():
            self.grad += (other * self.data**(other-1)) * out.grad
        out.back = b
        return out

    # -self
    def __neg__(self):
        return self * -1

    # other + self
    def __radd__(self, other): 
        return self + other

    # self - other
    def __sub__(self, other): 
        return self + (-other)

    # other - self
    def __rsub__(self, other):
        return other + (-self)

    # other * self
    def __rmul__(self, other):
        return self * other

    # self / other
    def __truediv__(self, other):
        return self * other**-1

    # other / self
    def __rtruediv__(self, other):
        return other * self**-1
