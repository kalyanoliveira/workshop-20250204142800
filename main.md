# Neural networks workshop

Build and train your very own neural network in Python, completely from scratch.

Kalyan Castro de Oliveira 

---

This presentation is entirely based on and inspired by [Andrej Karpathy](https://karpathy.ai/)'s *[The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)*. It is publicly available as a markdown file in INSERT LINK HERE, rendered by [Obsidian](https://obsidian.md/)'s internal implementation of [`reveal.js`](https://revealjs.com/).

---

> [!question]
> What is a neural network?

---

> [!question]
> What is a (machine learning's) neural network?

---

## What is a neural network?

> [!success] Answer
> Just a function.

---

## What is a neural network?

> [!question] 
> A function of what? Which outputs what?

---

## What is a neural network?

> [!example] Examples:
> *A function of $x$ named $f$, outputting one value:*
> $$f(x) = 3x^2 - 7x + 10$$
> *A function of $a$, $b$, and $c$ named $g$, outputting two values:*
> $$g(a, b, c) = (3ab  -c, ac + 7b)$$

---

## What is a neural network?

> [!question]
> A function of what? Which outputs what?

To answer this, let's explore **how to build neural networks**.

---

### Building a neural network

> [!question]
> What is a neuron?

---

#### A biologist's neuron

![Illustration of a biological neuron|700](https://i0.wp.com/post.healthline.com/wp-content/uploads/2022/01/1932990_An-Easy-Guide-To-Neurons-01.jpg?w=1155&h=2007)

---

#### A mathematician's neuron

[Let's write](https://excalidraw.com/#room=8a2611e882a298daa7a1,hfVBW4R87P0iEcGU3eMq0Q) that function!

---

#### A mathematician's neuron

![[20250114101600.png|700]]

---

##### Dimensionality

![[20250114110900.svg|700]]

---

#### Coding a neuron

> [!quote] [Andrej Karpathy](https://karpathy.ai/)
> Code is truth.

Get ready to code!

---

#### Coding a neuron

We want this kind of behavior:

```python
# `main.py`

from neuron import Neuron
d = 3
n = Neuron(d)

i = [1, 2, 3]
o = n(i)
print(o)
```

---

#### Coding a neuron

```python
# `neuron.py`

import numpy as np
np.random.seed(42)

class Neuron:

    def __init__(self, d):
        self.ws = [
            np.random.uniform(-1, 1) for _ in range(d)
        ]
        self.b = np.random.uniform(-1, 1)
        self.f = np.tanh

    def __call__(self, xs):
        return self.f(
            sum(
                [
                    x*w for x, w in zip(xs, self.ws)
                ] 
                + [self.b]
            )
        )
```

---

### Building a neural network

> [!question]
> What is a neuron?

---

### Building a neural network

> [!success] Answer
> ```python
> # Excerpt from `neuron.py`
>
> class Neuron:
> 
>     def __init__(self, d):
>         self.ws = [
>             np.random.uniform(-1, 1) for _ in range(d)
>         ]
>         self.b = np.random.uniform(-1, 1)
>         self.f = np.tanh
> 
>     def __call__(self, xs):
>         return self.f(
>             sum(
>                 [
>                     x*w for x, w in zip(xs, self.ws)
>                 ] 
>                 + [self.b]
>             )
>         )
> ```

---

### Building a neural network

> [!question] 
> What are networks of neurons?

---

### Building a neural network

> [!question] 
> What are networks of neurons?

To answer this, let's explore **how to chain neurons to create networks**.

---

#### Chaining neurons to create networks

[Let's chain some neurons!](https://excalidraw.com/#room=85272f64cdda49bd80eb,hnP3fhhPDIS6-jHSKl-GfA)

---

##### Layers

![[20250114113400.svg|500]]

---

##### Coding a layer

The desired behavior:

```python
# `main.py`

from layer import Layer
d = 3
nn = 4
l = Layer(d, nn)

i = [1, 2, 3]
o = l(i)
print(o)
```

---

##### Coding a layer

```python
# `layer.py`

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
```

---

##### MLP (a.k.a. "vanilla", "fully connected")

![[20250114112600.svg|700]]

---

##### Coding an MLP

Desired behavior is:

```python
# `main.py`

from mlp import MLP
ni = 3
nns = [2, 4, 4, 2]
mlp = MLP(ni, nns)

i = [1, 2, 3]
o = mlp(i)
print(o)
```

---

##### Coding an MLP

```python
# `mlp.py`

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
```

---

### Building a neural network

> [!question] 
> What are networks of neurons?

We said we were going to explore **how to chain neurons to create networks** to answer this.

Which we have now done.

---

### Building a neural network

> [!success] Answer
> ```python
> # Excerpt from `mlp.py`
>
> class MLP:
> 
>     def __init__(self, d, nns):
>         self.ls = [
>             Layer(d, nn) for d, nn 
>                 in zip(
>                     ([d] + nns)[:-1],
>                     nns
>                 )
>         ]
> 
>     def __call__(self, xs):
>         x = xs
>         for l in self.ls:
>             x = l(x)
>         return x
> ```

---

## What is a neural network?

> [!question]
> A function of what? Which outputs what?

We said we were going to explore **how to build neural networks** to answer this.

Which we have now done.

---

## What is a neural network?

We can rephrase

```python
# Excerpt of `main.py`

i = [1, 2, 3]
o = mlp(i)
```

into

```python
i = [x, y, z]
o = mlp(i)
```

which allows us to say

$$
MLP(x, y, z) = \text{???}
$$

---

## What is a neural network?

> [!question]
> A function of what?

---

## What is a neural network?

> [!success] Answer
> $$MLP(x, y, z)$$

---

## What is a neural network?

> [!question]
> Which outputs what?

$$MLP(x, y, z) = \text{???}$$

---

## What is a neural network?

We can rephrase

```python
i = [x, y, z]
o = mlp(i)
```

into

```python
i = [x, y, z]
o = mlp.__call__(i)
```

and, remember,

```python
# Excerpt of `mlp.py`

    def __call__(self, xs):
        x = xs
        for l in self.ls:
            x = l(x)
        return x
```

Also remember that, in our example,

```python
# Excerpt of `main.py`

ni = 3
nns = [2, 4, 4, 2]
mlp = MLP(ni, nns)
```

Using all that knowledge, we can find an expression for the $\text{???}$ of $MLP(x, y, z) = \text{???}$

---

## What is a neural network?

> [!success] Answer
> ![[AnimateTyping_ManimCE_v0.18.1.gif]]

---

## What is a neural networks? 

1.

> [!question]
> What is a (machine learning's) neural network?

---

## What is a neural network?

2.

> [!success] Answer
> Just a function

---

## What is a neural network?

3.

> [!question] 
> A function of what? Which outputs what?

---

## What is a neural network?

4.

> [!success] Answer
> $$MLP(x, y, z)$$

> [!success] Answer
> ![[AnimateTyping_ManimCE_v0.18.1.gif|500]]

---

## What is a neural network?

> [!abstract] TL;DR
> Neural network (machine learning): a function of however many inputs we choose, to however many outputs we choose.

---

> [!success] Answers...?
> $$MLP(x, y, z)$$
> ![[AnimateTyping_ManimCE_v0.18.1.gif|500]]

---

> [!question]
> Are these really the answers we wanted?

---

> [!question]
> Where's the **intuition**?

---

> [!question]
> What do neural networks do?

---

## What do neural networks do?

> [!question]
> What do neural networks do?

To answer this, let's **first** answer 

> [!question]
> Where's the intuition in  $MLP(x, y, z)$ and 
>
> ![[AnimateTyping_ManimCE_v0.18.1.gif|500]]
> ?

And to answer that, let's **rethink what neurons and layers do**.

---

### A shift in perspective: representations in neurons

Neurons generate representations.

![[20250202193600.png|900]]

---

### A shift in perspective: representations in layers

Layers generate representations.

![[20250202202505.png|900]]

---

### A shift in perspective: representations

> [!question]
> What do these representations actually look like?

---

#### Random parameters

> [!success] Answer
> Completely random!
> 
> Remember: weights and biases are random by default!
> 
> ```python
> # Excerpt from `neuron.py`
> 
>         self.ws = [
>             np.random.uniform(-1, 1) for _ in range(d)
>         ]
>         self.b = np.random.uniform(-1, 1)
>         self.f = np.tanh
> ```
> 
> ```python
> # Excerpt from `main.py`
> 
> i = [1, 2, 3]
> o = mlp(i)
> print(o)
> ```
> 
> ```
> [user@hostname:~]$ python main.py
> [np.float64(-0.9157536075966881), np.float64(-0.2499947804027186)]
> ```

---

## What do neural networks do?

> [!question]
> Where's the intuition in  $MLP(x, y, z)$ and
>
> ![[AnimateTyping_ManimCE_v0.18.1.gif|500]]
> ?

We said we were going to **rethink what neurons and layers do** to answer this.

Which we have now done.

---

## What do neural networks do?

> [!success] Answer
> In the perspective shift whereby we view generating (albeit *random*) representations as the core function of neurons, layers, and, by extension, networks.

---

## What do neural networks do?

> [!question]
> What do neural networks do?

We said we were going to answer this by **first** answering where intuition was addressed in neural networks.

Which we have now done.

---

## What do neural networks do?

> [!failure]
> But we still can't answer it.

---

## What do neural networks do?

> [!tip] Insight
> What if these representations were *not* random?

---

### The potential - in neurons

> [!tip] Insight
> What if we manually controlled weights and biases, generating intentional (rather than random) representations?
>
> ![[20250203234505.png|700]]

---

### The potential - in layers

> [!tip] Insight
> What if we manually controlled weights and biases, generating intentional (rather than random) representations?
> 
> ![[20250203234705.png|700]]

---

### The potential

> [!question]
> *Manually*?

---

### The potential - extended

> [!tip] Insight
> What if we *automatically* controlled weights and biases, generating intentional (rather than random) representations?

---

### The potential - extended

> [!tip] Insight
> This is why we *train* neural networks!
> 
> We want them to *learn* how to control their own weights and biases to generate intentional representations.
> 
> ![[20250203230605.png|500]]

---

## What do neural networks do?

> [!question]
> What do neural networks do?

---

## What do neural networks do?

> [!success] Answer
> They *learn* to output what we want by being *trained* on inputs and their desired outputs.

---

> [!question]
> How do we "train" neural networks?

---

## Training networks

> [!success] Answer
> 1. Collect data that represents something, generating **inputs**.
> 2. Pair up **inputs** with **desired** output data, generating **ground truth outputs**.
> 3. Collect the **outputs** that you get by giving the network the generated **inputs**, generating **predicted outputs**.
> 4. Calculate the **difference** between the **ground truth outputs** and the **predicted outputs**.
> 5. Slightly change parameters (weights and biases) to reduce **difference**.
> 6. Repeat from 3.

---

### Difference


> [!question]
> > 4. Calculate the **difference** between the **ground truth outputs** and the **predicted outputs**.
>
> How could we achieve this?

---

### Loss

A single value that characterizes how "bad" our network is doing.

`loss == difference`


---

###  The loss function

`loss = loss_f(po, gto)`

---

### The loss function

The kind of behavior that we want:

```python
# `main.py`

from mlp import MLP
ni = 3
nns = [2, 4, 4, 2]
mlp = MLP(ni, nns)

i = [1, 2, 3]
gto = [1, -1]

po = mlp(i)

from loss import sse
loss_f = sse 

loss = loss_f(po, gto)
print(loss)
```

---

### The loss function

$N$ is the number of outputs of our network:

$$
\text{loss} = \sum_{i = 1}^{N} (p_i- gt_i)^2
$$

This is also known as Sum of Squared Errors (SSE).

```python
# Excerpt from `loss.py`

def sse(po, gto):
    return sum(
        [
            (p - gt)**2 for p, gt in zip (po, gto)
        ]
    )
```

---

## Training networks


> [!question]
> > 4. Calculate the **difference** between the **ground truth outputs** and the **predicted outputs**.
>
> How could we achieve this?

---

## Training networks

> [!success] Answer
> Loss function.

---

## Training networks


> [!question]
> > 5. Slightly change parameters (weights and biases) to reduce **difference**.
>
> How could we achieve this?

---

### The loss function is a function of parameters

> [!tip] Insight
> The loss function is a function of weights and biases (i.e., parameters).

---

### The loss function is a function of parameters

Recall that

```python
# Rearranged excerpts from `main.py`

i = [1, 2, 3]
po = mlp(i)

gto = [1, -1]

loss_f = sse
loss = loss_f(po, gto)
```

Thus, we can rephrase

```python
loss = sse(
    mlp(
        [x, y, z]
    ), 
    gt0, gt1
)
```

allowing us to say

$$
\text{LOSS}(x, y, z, w_{000}, w_{001}, ..., b_{13}, gt_0, gt_1) = \text{???}
$$

Also, recall that

```python
# Excerpt from `loss.py`

def sse(po, gto):
    return sum(
        [
            (p - gt)**2 for p, gt in zip (po, gto)
        ]
    )
```

Using this information, can we write an expression for the loss function?

---

### The loss function is a function of parameters

> [!success] Answer
> Of course!
>
> ![[AnimateTyping_ManimCE_v0.19.0-2.gif]]

---

## Training networks

Still, though,

#
> [!question]
> > 5. Slightly change parameters (weights and biases) to reduce **difference**.
>
> How could we achieve this?

---

## Training networks

> [!tip] Insight
> If we could find the influence of each weight and bias (i.e., parameter) in our loss, we could use that knowledge to slightly change them, slightly reducing our loss!

---

### The influence and derivatives

> [!tip] Insight
> The phrase
>
> > \[...] the influence of each weight and bias \[...] in our loss \[...]
>
> sounds like
> 
> > the influence of variables in a function.

---

### The influence and derivatives

> [!tip] Insight
> > The influence of variables in a function
>
> sounds like derivatives!

---

#### Derivatives 101

> [!question]
> What is the derivative of 
> $$
> f(x) = 3x^2 - 7x + 10
> $$ 
> with respect to $x$ at $x =3$?

---

#### Derivatives 101

> [!success] Answer...?
> $$
> f'(x) = 6x - 7 \implies f(3) = 11
> $$
>

---

#### Derivatives 101

> [!question]
> No, *really*, **what is** the derivative of 
> $$
> f(x) = 3x^2 - 7x + 10
> $$ 
> with respect to $x$ at $x =3$?

---

#### Derivatives 101

![[20250203150805.png|800]]

---

#### Derivatives 101

![[20250203151505.png|700]]

---

#### Derivatives 101

> [!question]
> No, *really*, **what is** the derivative of 
> $$
> f(x) = 3x^2 - 7x + 10
> $$ 
> with respect to $x$ at $x =3$?

---

#### Derivatives 101

>[!success] Answer
> ![[20250203151505.png|500]]

---

#### Leveraging derivatives

> [!tip] Insight
> Find the derivative of each variable of a function with respect to that function to slightly change each variable, slightly reducing that function!

---

#### Leveraging derivatives

> [!example]
> Simple example:
> 
> ```python
> # `derivative0.py`
> 
> def f(x):
>     return 3*x**2 - 7*x + 10
> 
> x = 3
> 
> h = 0.000001
> grad = (f(x + h) - f(x)) / h
> 
> print("Before:")
> print(f"{f(x)=}")
> 
> lr = 0.01
> x += lr * (-grad)
> 
> print("After:")
> print(f"{f(x)=}")
> ```
> 
> ```
> [user@hostname:~]$ python derivative0.py
> Before:
> f(x)=16
> After:
> f(x)=14.826299689521605
> ```

---

#### Leveraging derivatives

> [!example]
> A slightly more complex example:
> 
> ```python
> # `derivative1.py`
> 
> def g(a, b, c):
>     return 3*a*b - c
> 
> a = 1
> b = 2
> c = 3
> 
> h = 0.000001
> a_grad = (g(a + h, b, c) - g(a, b, c)) / h
> b_grad = (g(a, b + h, c) - g(a, b, c)) / h
> c_grad = (g(a, b, c + h) - g(a, b, c)) / h
> 
> print("Before:")
> print(f"{g(a, b, c)=}")
> 
> lr = 0.01
> a += lr * (-a_grad)
> b += lr * (-b_grad)
> c += lr * (-c_grad)
> 
> print("After:")
> print(f"{g(a, b, c)=}")
> ```
> 
> ```
> [user@hostname:~]$ python derivative1.py
> Before:
> g(a, b, c)=3
> After:
> g(a, b, c)=2.5454000000421946
> ```

---

#### Leveraging derivatives

> [!question]
> Could we do something similar to these examples with our loss function?

---

#### Leveraging derivatives

> [!failure] Answer
> No
>
> Not only is the following **not** actual code, it
> - does not follow our current implementation of `MLP`,
> - it is too computationally expensive, and
> - it's too impractical to write!
> 
> > [!example]
> > ```python
> > # `derivative2.py`
> > 
> > # This is not actual Python code!
> > 
> > import numpy as np
> > 
> > def mlp(x, y, z, 
> >     w000, w100, w200, b00
> >     w010, w110, w210, b10
> >     ...
> >     ...
> >     ...
> >     w013, w113, w213, w313, b13 
> > ):
> >     ...
> >     ...
> >     ...
> >     return po
> > 
> > i = [1, 2, 3]
> > 
> > w000 = np.random.uniform(-1, 1)
> > w100 = np.random.uniform(-1, 1)
> > ...
> > ...
> > ...
> > b13 = np.random.uniform(-1, 1)
> > 
> > gto = [1, -1]
> > 
> > from loss import sse
> > loss_f = sse
> > 
> > h = 0.000001
> > w000_grad = (
> >     (
> >         loss_f(
> >             mlp(x, y, z, w000 + h, ..., b13), 
> >             gto
> >         )
> >     ) 
> >     - 
> >     (
> >         loss_f(
> >             mlp(x, y, z, w000, ..., b13), 
> >             gto
> >         )
> >     )
> > ) / h
> > w000_grad = (
> >     (
> >         loss_f(
> >             mlp(x, y, z, w000, w100 + h, ..., b13), 
> >             gto
> >         )
> >     ) 
> >     - 
> >     (
> >         loss_f(
> >             mlp(x, y, z, w000, w100, ..., b13), 
> >             gto
> >         )
> >     )
> > ) / h
> > ...
> > ...
> > ...
> > b13_grad = (
> >     (
> >         loss_f(
> >             mlp(x, y, z, w000, ..., b13 + h),
> >             gto
> >         )
> >     )
> >     -
> >     (
> >         loss_f(
> >             mlp(x, y, z, w000, ..., b13),
> >             gto
> >         )
> >     )
> > ) / h
> > 
> > print("Before:")
> > print(f"{loss_f(x, y, z, w000, ..., b13)=}")
> > 
> > lr = 0.01
> > 
> > w000 += lr * (-w000_grad)
> > w100 += lr * (-w100_grad)
> > ...
> > ...
> > ...
> > b13 += lr * (-b13_grad)
> > 
> > print("After:")
> > print(f"{loss_f(x, y, z, w000, ..., b13)=}")
> > ```

---

## Training networks

We still could not answer


> [!question]
> > 5. Slightly change parameters (weights and biases) to reduce **difference**.
>
> How could we achieve this?

But we are getting closer. Just a couple more insights.

---

## Training networks

> [!tip] Insight
> As it turns out, there's a different, better way (for our situation) of calculating derivatives.
> And it's called...

---

## Training networks

> [!tip] Insight
> Backpropagation.

---

### Backpropagation

> [!question]
> What is backpropagation?

---

### Backpropagation

![[20250204013005.png|800]]

---

### Backpropagation

![[20250204013305.png|800]]

---

### Backpropagation

![[20250204013505.png|800]]

---

### Backpropagation

![[20250204013605.png|800]]

---

### Backpropagation

![[20250204013805.png|800]]

---

### Backpropagation

![[20250204013905.png|800]]

---

### Backpropagation

![[20250204014005.png|800]]

---

### Backpropagation

![[20250204014005.png|800]]

---

### Backpropagation

![[20250204014105.png|800]]

---

#### Operations and flow of gradients

![[20250204015205.png|800]]

---

#### Coding backpropagation

Let's code backpropagation!

---

##### `Value`

```python
# `value.py`
class Value:

    def __init__(
        self,
        data,
    ):
        self.data = data

    def __repr__(self):
        return f"Value({self.data=})"
```

---

##### Adding and multiplying `Value`s

Desired behavior:

```python
# `main.py`

from value import Value

a = Value(2)
b = Value(-3)
c = Value(10)
d = a*b
e = d + c
f = Value(-2)
L = e*f

print(L)
```

---

##### Adding and multiplying `Value`s

```python
# Excerpt from `value.py`

    def __add__(self, other):
        return Value(
            self.data + other.data
        )

    def __mul__(self, other):
        return Value(
            self.data * other.data
        )
```

---

##### Keeping track of expressions and gradients

```python
# Excerpts from `value.py`

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

    def __add__(self, other):
        return Value(
            self.data + other.data,
            (self, other),
            "+"
        )

    def __mul__(self, other):
        return Value(
            self.data * other.data,
            (self, other),
            "*"
        )
```

---

##### Keeping track of expressions and gradients

```python
# `main.py`

from value import Value

a = Value(2)
b = Value(-3)
c = Value(10)
d = a*b
e = d + c
f = Value(-2)
L = e*f

print(L, L.prev, L.op)
```

```
[user@hostname:~]$ python main.py
Value(self.data=-8) {Value(self.data=4), Value(self.data=-2)} *
```

---

##### Diagramming expressions

Use `draw.py`

```python
# `main.py`

from value import Value

a = Value(2, label="a")
b = Value(-3, label="b")
c = Value(10, label="c")
d = a*b; d.label = "d"
e = d + c; e.label = "e"
f = Value(-2, label="f")
L = e*f; L.label = "L"

from draw import draw
draw(L)
```

![[out1.svg]]

---

##### Extending functionality - constants

```python
# Excerpt from `value.py`

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data + other.data,
            (self, other),
            "+"
        )

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            self.data * other.data,
            (self, other),
            "*"
        )
```

---

##### Extending functionality - `tanh`

```python
# Excerpts from `value.py`

import numpy as np

    def tanh(self):
        return Value(
            np.tanh(self.data), 
            (self, ), 
            "tanh"
        )

```

---

##### Extending functionality - `pow`

```python
# Excerpt from `value.py`

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        return Value(
            self.data**other, 
            (self, ), 
            f"**{other}"
        )
````

---

##### Extending functionality - derived operations

```python
# Excerpt from `value.py`

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
```

---

##### Implementing the backward pass

> [!tip]
> Make parent nodes update the `grad` of `prev` nodes by giving parent nodes a `backward()`, which updates the grad of its `prev` nodes.

```python
# Excerpt of `main.py`

L.backward()
```

---

###### Almost fully automated backward pass

> [!tip]
> `back()` for updating the grad of just the immediate `prev` nodes

```python
# Excerpt of `value.py`

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
```

---

###### Almost fully automated backward pass

```python
# Excerpt from `value.py`

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
```

---

###### Almost fully automated backward pass

```python
# `main.py`

from value import Value

a = Value(2, label="a")
b = Value(-3, label="b")
c = Value(10, label="c")
d = a*b
d.label = "d"
e = d + c
e.label = "e"
f = Value(-2, label="f")
L = e*f
L.label = "L"

L.grad = 1
L.back()
e.back()
d.back()

from draw import draw
draw(L)
```

![[out2.svg]]

---

###### Fully automated backward pass

```python
# Excerpt from `value.py`

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
```

```python
# `main.py`

from value import Value

a = Value(2, label="a")
b = Value(-3, label="b")
c = Value(10, label="c")
d = a*b
d.label = "d"
e = d + c
e.label = "e"
f = Value(-2, label="f")
L = e*f
L.label = "L"

L.backward()

from draw import draw
draw(L)
```

![[out2.svg|500]]

---

## Training networks

> [!question]
> > 5. Slightly change parameters (weights and biases) to reduce **difference**.
>
> How could we achieve this?

---

## Training networks

> [!success] Answer
> 1. Perform backpropagation on `loss` and
> 2. slightly change each parameter in the opposite direction of its gradient.

This requires us to wrap our existing neural network code in `Value` objects, and some other things.

Let's do this!

---

### The training loop

The kind of behavior that we want:

```python
# `main.py`

from mlp import MLP

ni = 3
nns = [2, 4, 4, 2]
mlp = MLP(ni, nns)

i = [1, 2, 3]
gto = [1, -1]

from loss import sse
loss_f = sse

k = 100
for _ in range(k):
    po = mlp(i)

    loss = loss_f(po, gto)
    print(f"{loss=}")

    for p in mlp.ps():
        p.grad = 0

    loss.backward()

    lr = 0.01
    for p in mlp.ps():
        p.data += lr * (-p.grad)
```

---

### `mlp.ps()`

```python
# Excerpt of `neuron.py`

    def ps(self):
        return self.ws + [self.b]
```

```python
# Excerpt of `layer.py`

    def ps(self):
        return [p for n in self.ns for p in n.ps()]
```

```python
# Excerpt of `mlp.py`

    def ps(self):
        return [p for l in self.ls for p in l.ps()]
```

---

### Wrapping in `Value`

```python
# Excerpts from `neuron.py`

from value import Value

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
```

---

### Training for multiple examples

```python
# Excerpt from `loss.py`

def total_sse(pos, gtos):
    return sum(
        [
            sse(po, gto) for po, gto in zip(pos, gtos)
        ]
    )
```

```python
# `main.py`

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
````
