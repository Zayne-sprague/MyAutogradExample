from autograd.Functions import *
import numpy as np
import uuid

class Zensor:

    value: np.ndarray
    id: uuid.UUID
    requires_grad: bool

    def __init__(self, data, requires_grad=True, _id=None, symbol=None, **kwargs):
        if _id:
            self.id = _id
        else:
            self.id = uuid.uuid4()

        if isinstance(data, np.ndarray):
            self.value = data
        else:
            if not isinstance(data, list):
                data = [data]
            self.value = np.array(data, **kwargs)

        self.is_zensor = True
        self.symbol = symbol


        self.graph = None
        self.grad = None
        self.computed_backward = False
        self.require_grad = requires_grad


    def float(self):
        self.value = self.value.astype(float)
        return self

    def transpose(self, axis=None):
        output = Transpose()(self, axis=axis)
        return output

    def sum(self, axis=None):
        output = Sum()(self, axis=axis)
        return output

    def mean(self):
        output = self.sum()
        divisor = self.value.shape[0]

        return output / divisor

    def log(self):
        output = Log()(self)
        return output

    def exp(self):
        output = Exp()(self)
        return output

    def __mul__(self, other):
        output = Mult()(self, other)
        return output

    def __matmul__(self, other):
        output = MatMult()(self, other)
        return output

    def __add__(self, other):
        output = Add()(self, other)
        return output

    def __iadd__(self, other):
        output = Add()(self, Zensor(other))
        return output

    def __sub__(self, other):
        output = Subt()(self, other)
        return output

    def __idiv__(self, other):
        output = Div()(self, other)
        return output

    def __truediv__(self, other):
        output = Div()(self, other)
        return output

    def __neg__(self):
        output = Neg()(self)
        return output

    def __getitem__(self, item):
        output = Getitem()(self, item)
        return output

    def __str__(self):
        if self.symbol:
            return f'{self.symbol}'
        return f'Zensor[{self.value}]'

    def __eq__(self, other):
        if isinstance(other, tuple) or isinstance(other, list):
            return Zensor(self.value == other)
        else:
            return other.value == self.value

    def zero_grad(self):
        self.grad = None
        self.graph = None
        self.computed_backward = False
        return self

    def backward(self, grad=None):
        if not self.graph:
            return

        if not isinstance(grad, np.ndarray) and grad == None:
            self.grad = np.array([1.])

        self.computed_backward = True



        if len(self.graph.ins) > 1:
            grads = self.graph.grad_fn(self.grad, self.graph.ins[0].value, self.graph.ins[1].value)
        else:
            grads = self.graph.grad_fn(self.grad, self.graph.ins[0].value)

        for x, grad in zip(self.graph.ins, grads):
            if not isinstance(grad, np.ndarray) and grad == None:
                x.grad += grad.copy()
            else:
                x.grad = grad.copy()


        for x, grad in zip(self.graph.ins, grads):
            if not x.computed_backward:
                x.backward(grad.copy())

def zSigmoid(z):
    output = Sigmoid()(z)
    return output

def zArgmax(z, axis=None):
    return Zensor(np.argmax(z.value, axis=axis))

