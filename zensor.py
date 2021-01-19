from computation_node import ComputationNode
from gradient_funcs import *
import numpy as np
import uuid

class Zensor:

    def __init__(self, data, **kwargs):
        if isinstance(data, np.ndarray):
            self.value = data
        else:
            if not isinstance(data, list):
                data = [data]
            self.value = np.array(data, **kwargs)

        self.g = []

        self.grad = np.zeros_like(data, dtype=float)

        self.id = uuid.uuid4()

        self.require_grad = True

    def __mul__(self, other):
        output = self.value * other.value
        output = Zensor(output)

        if self.require_grad:
            output.g = [
                ComputationNode(self, mult(other.value), op="mul"),
                ComputationNode(other, mult(self.value), op="mul"),
            ]

        return output

    def __matmul__(self, other):
        output = Zensor(np.dot(self.value, other.value))

        if self.require_grad:
            output.g = [
               ComputationNode(self, mat_mult_1(other.value), op='matmul0'),
               ComputationNode(other, mat_mult_0(self.value), op='matmul1'),
            ]

        return output

    def __add__(self, other):
        output = Zensor(self.value + other.value)

        if self.require_grad:
            if self.value.shape[1] > 1 and (len(other.value.shape) == 1 or other.value.shape[1] == 1):
                output.g = [
                    ComputationNode(self, add(other.value), 'add'),
                    ComputationNode(other, vadd(self.value), 'vadd')
                ]
            else:
                output.g = [
                    ComputationNode(self, add(other.value), 'add'),
                    ComputationNode(other, add(self.value), 'add')
                ]

        return output

    def __iadd__(self, other):
        output = Zensor(self.value + other)

        if self.require_grad:
            output.g = [
                ComputationNode(self, add(other), 'iadd'),
                ComputationNode(Zensor(other), add(self.value), 'iadd')
            ]

        return output

    def __sub__(self, other):
        output = Zensor(self.value - other.value)

        if self.require_grad:
            if self.value.shape[1] > 1 and other.value.shape[1] == 1:
                output.g = [
                    ComputationNode(self, sub_0(other.value), 'sub0'),
                    ComputationNode(other, v_sub_1(self.value), 'v_sub_1')
                ]
            else:
                output.g = [
                    ComputationNode(self, sub_0(other.value), 'sub0'),
                    ComputationNode(other, sub_1(self.value), 'sub1')
                ]

        return output

    def __idiv__(self, other):
        output = Zensor(self.value / other.value)

        if self.require_grad:
            output.g = [
                ComputationNode(self, div_0(other.value), op='div0'),
                ComputationNode(self, div_1(self.value, other.value), op='div1')
            ]

        return output

    def __truediv__(self, other):
        if other == 0:
            output = Zensor(0)
        else:
            output = Zensor(self.value / other)

        if self.require_grad:
            output.g = [
                ComputationNode(self, div_0(other), 'div0')
            ]

        return output



    def __neg__(self):
        output = Zensor(-self.value)

        if self.require_grad:
            output.g = [
                ComputationNode(self, neg(self.value), 'neg')
            ]

        return output

    def __str__(self):
        return f'Tensor[{self.value}]'

    def __getitem__(self, item):
        selected = Zensor(self.value[item])
        selected.grad = -np.ones_like(self.grad)
        selected.grad[item] = 0
        selected.g = self.g
        return selected

    def __eq__(self, other):
        if isinstance(other, tuple) or isinstance(other, list):
            return Zensor(self.value == other)
        else:
            return other.value == self.value

    def squeeze(self):
        self.value = self.value.squeeze()
        return self

    def unsqueeze(self, axis=0):
        self.value = np.expand_dims(self.value, axis=axis)
        return self

    def sum(self, axis=None):
        if axis:
            output = Zensor(self.value.sum(axis))
        else:
            output = Zensor(self.value.sum())

        if self.require_grad:
            cloned = 1
            if len(self.value.shape) > 1:
                cloned = self.value.shape[axis]

            output.g = [
                ComputationNode(self, sum(self.value, cloned), op='sum')
            ]


        return output

    def mean(self):
        output = self.sum()
        divisor = self.shape(0)[0]

        return output / divisor

    def log(self):
        output = Zensor(np.log(self.value))

        if self.require_grad:
            output.g = [
                ComputationNode(self, log(self.value, output.value), op='log')
            ]

        return output

    def exp(self):
        output = Zensor(np.exp(self.value))

        if self.require_grad:
            output.g = [
                ComputationNode(self, exp(self.value, output.value), op='exp')
            ]

        return output

    def float(self):
        self.value = self.value.astype(float)
        return self

    def zero_grad(self):
        self.grad = np.zeros_like(self.value, dtype=float)
        self.g = []
        return self

    def shape(self, axis=None):
        if axis:
            return self.value.shape[axis][0]
        return self.value.shape

    def backward(self):
        self.grad = np.where(self.grad == 0, 1, self.grad)
        self.grad = np.where(self.grad == -1, 0, self.grad)
        return self.__backward__()

    def __backward__(self):
        if len(self.g) == 0:
            return 0

        node_1 = self.g[0]

        try:
            node_1.input_node.grad = node_1.input_node.grad + node_1.grad_func(self.grad)
        except Exception as e:
            print(e)

        if len(self.g) > 1:
            node_2 = self.g[1]
            try:
                node_2.input_node.grad = node_2.input_node.grad + node_2.grad_func(self.grad)
            except Exception as e:
                print(e)

            node_1.input_node.__backward__()
            node_2.input_node.__backward__()
        else:
            node_1.input_node.__backward__()

def zensor_Sigmoid(z):
    v = z.value
    y = Zensor(1 / (1 + np.exp(-v)) )

    y.g = [
        ComputationNode(z, sigmoid(z.value), op='sigmoid')
    ]

    return y

def zensor_exp(z):
    y = Zensor(np.exp(z.value))
    y.g = [
        ComputationNode(z, exp, 'zexp')
    ]
    return y

def zensor_argmax(z, axis):
    return Zensor(np.argmax(z.value, axis=axis))