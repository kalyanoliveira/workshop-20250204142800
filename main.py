from mlp import MLP

ni = 3
nns = [2, 4, 4, 2]
mlp = MLP(ni, nns)

inputs = [
    [1, 2, 3],
    [3, -2, 1],
    [1, 0, 1],
    [2, 2, -1]
]
gtos = [
    [1, -1],
    [-1, 1],
    [0.5, 1],
    [-1, -0.5],
]

from loss import total_sse
loss_f = total_sse

k = 1000
for _ in range(k):

    pos = [mlp(i) for i in inputs]

    loss = loss_f(pos, gtos)
    print(f"{loss=}")

    for p in mlp.ps():
        p.grad = 0

    loss.backward()

    lr = 0.01
    for p in mlp.ps():
        p.data += lr * (-p.grad)

