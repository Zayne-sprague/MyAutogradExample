import numpy as np
from autograd import Function
from autograd import GraphPartial

# TODO - separate these out into a more manageable format

class Getitem(Function):
    def __init__(self):
        super().__init__("get_item")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        idx = args[1]

        out = self.__creator__(arg0.value[idx])

        if isinstance(idx, list) and (len(idx) == 0 or not isinstance(idx[0], slice)):
            idx = np.array(idx, dtype='int64')

        out.graph = GraphPartial([arg0], self.name, out, lambda g, a0: self.grad_fn(g, a0, idx))

        return out

    @staticmethod
    def grad_fn(g, a0, idx):
        out = np.zeros_like(a0)
        out[idx] = g
        return (out,)


class Add(Function):

    def __init__(self):
        super().__init__("add")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        arg1 = args[1]

        out = self.__creator__(arg0.value + arg1.value)
        out.graph = GraphPartial([arg0, arg1], self.name, out, lambda g, a0, a1: (g, g))

        return out


class Mult(Function):

    def __init__(self):
        super().__init__("mult")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        arg1 = args[1]

        out = self.__creator__(arg0.value * arg1.value)
        out.graph = GraphPartial([arg0, arg1], self.name, out, lambda g, a0, a1: (a1 * g, a0 * g))

        return out


class Subt(Function):

    def __init__(self):
        super().__init__("subt")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        arg1 = args[1]

        out = self.__creator__(arg0.value - arg1.value)
        out.graph = GraphPartial([arg0, arg1], self.name, out, lambda g, a0, a1: (g, -g))

        return out


class Div(Function):

    def __init__(self):
        super().__init__("div")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        arg1 = args[1]

        out = self.__creator__(arg0.value / arg1.value)
        out.graph = GraphPartial([arg0, arg1], self.name, out, lambda g, a0, a1: (g / a1, (-g * a0)/( a1 ** 2)))

        return out


class Exp(Function):
    def __init__(self):
        super().__init__("exp")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]

        out = self.__creator__(np.exp(arg0.value))
        out.graph = GraphPartial([arg0], self.name, out, lambda g, a0 : self.grad_fn(g, a0, out.value))

        return out

    @staticmethod
    def grad_fn(g, a0, ans):
        return (ans * g,)

class Log(Function):
    def __init__(self):
        super().__init__("log")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]

        out = self.__creator__(np.log(arg0.value))
        out.graph = GraphPartial([arg0 ], self.name, out, lambda g, a0: (g / a0,))

        return out

class Neg(Function):
    def __init__(self):
        super().__init__("neg")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]

        out = self.__creator__(-arg0.value)
        out.graph = GraphPartial([arg0 ], self.name, out, lambda g, a0: (-g,))

        return out

class Sqr(Function):
    def __init__(self):
        super().__init__("sqr")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]

        out = self.__creator__(arg0.value * arg0.value)
        out.graph = GraphPartial([arg0], self.name, out, lambda g, a0: (2 * a0 * g,))

        return out


class MatMult(Function):
    def __init__(self):
        super().__init__("mat_mult")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        arg1 = args[1]

        out = self.__creator__(np.dot(arg0.value, arg1.value))
        out.graph = GraphPartial([arg0, arg1], self.name, out, self.grad_fn)

        return out

    @staticmethod
    def grad_fn(g, a0, a1):
        if len(g.shape) == 1 and len(g) == 1:
            a0_res = g * a1.transpose()
            a1_res = g * a0.transpose()
        else:
            a0_res = np.matmul(g, a1.transpose())
            a1_res = np.matmul(a0.transpose(), g)
        return a0_res, a1_res


class Transpose(Function):
    def __init__(self):
        super().__init__("transpose")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        axis = kwargs.get('axis', None)

        out = self.__creator__(np.transpose(arg0.value, axis))
        out.graph = GraphPartial([arg0], self.name, out, lambda g, a0: self.grad_fn(g, a0, axis))

        return out

    @staticmethod
    def grad_fn(g, a0, axis):
        if axis is not None:
            axis = np.argsort(axis)

        return np.transpose(g, axis)


class Sum(Function):
    def __init__(self):
        super().__init__("sum")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        axis = kwargs.get('axis', 0) or 0

        out = self.__creator__(arg0.value.sum(axis))
        out.graph = GraphPartial([arg0], self.name, out, lambda g, a0: self.grad_fn(g, a0, arg0.value.shape[axis]))

        return out

    @staticmethod
    def grad_fn(g, a0, shape):
        return (np.array([g.tolist(), ] * shape),)


class Sigmoid(Function):
    def __init__(self):
        super().__init__("sum")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]

        v = arg0.value
        out = self.__creator__(1 / (1 + np.exp(-v)))
        out.graph = GraphPartial([arg0], self.name, out, self.grad_fn)

        return out

    @staticmethod
    def grad_fn(g, a0):
        return (g * (np.exp(-a0) / (np.square(1.0 + np.exp(-a0)))),)


# Taken from the original Autograd code
def __metadata__(x):
    return np.shape(x), np.ndim(x), np.result_type(x), np.iscomplexobj(x)

