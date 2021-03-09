import numpy as np
from autograd import Function
from autograd import GraphPartial
from .util import unbroadcast

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
        out.graph = GraphPartial([arg0 ], self.name, out, self.grad_fn)

        return out

    @staticmethod
    def grad_fn(g, a0):
        # if np.ndim(g) == 1:
        #     g = np.expand_dims(g, 1)
        # if np.ndim(a0) == 1:
        #     a0 = np.expand_dims(a0, 1)
        return g / a0

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

class Sqrt(Function):
    def __init__(self):
        super().__init__("sqr")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]

        out = self.__creator__(np.sqrt(arg0.value))
        out.graph = GraphPartial([arg0], self.name, out, lambda g, a0: (g * 0.5 * a0 ** -0.5))

        return out


class MatMult(Function):
    def __init__(self):
        super().__init__("mat_mult")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]
        arg1 = args[1]

        out = self.__creator__(np.matmul(arg0.value, arg1.value))
        out.graph = GraphPartial([arg0, arg1], self.name, out, self.grad_fn)

        return out

    @staticmethod
    def grad_fn(g, a0, a1):
        a0_meta = __metadata__(a0)
        a1_meta = __metadata__(a1)
        g_meta = __metadata__(g)

        a0_shape, a0_ndim, _, _ = a0_meta
        a1_shape, a1_ndim, _, _ = a1_meta
        g_shape, g_ndim, _, _ = g_meta

        def a0_result(a0, a1, g):
            if g_ndim == 0:
                return g * a1

            if a0_ndim == 1:
                g = np.expand_dims(g, g_ndim - 1)
            if a1_ndim == 1:
                a1 = np.expand_dims(a1, 0)
                g = np.expand_dims(g, g_ndim)
            else:
                a1 = np.swapaxes(a1, a1_ndim - 2, a1_ndim - 1)

            return np.matmul(g, a1)

        def a1_result(a0, a1, g):
            if g_ndim == 0:
                return g * a0

            if a1_ndim == 1:
                g = np.expand_dims(g, g_ndim)
            if a0_ndim == 1:
                a0 = np.expand_dims(a0, 1)
                g = np.expand_dims(g, g_ndim - 1)
            elif a0_ndim == 2:
                a0 = np.swapaxes(a0, a0_ndim - 2, a0_ndim - 1)

            out = np.matmul(a0, g)
            if a1_ndim == 1:
                out = out.squeeze(g_ndim - 1)
            return out


        return (
                    unbroadcast(a0_result(a0, a1, g), a0_meta),
                    unbroadcast(a1_result(a0, a1, g), a1_meta)
                )



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
        # return (np.array([g.tolist(), ] * shape),)
        return (np.ones_like(a0) * g, )


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


class ReLU(Function):
    def __init__(self):
        super().__init__("sum")

    def __call__(self, *args, **kwargs):
        arg0 = args[0]

        v = arg0.value
        # mask =np.ma.masked_less_equal(v, 0)
        mask = np.ma.masked_greater(v, 0)
        v[v<=0] = 0
        out = self.__creator__(v)
        out.graph = GraphPartial([arg0], self.name, out, lambda g, a0 : self.grad_fn(g, a0, mask))

        return out

    @staticmethod
    def grad_fn(g, a0, mask):
        return ((mask * g).filled(0),)

# Taken from the original Autograd code
def __metadata__(x):
    return np.shape(x), np.ndim(x), np.result_type(x), np.iscomplexobj(x)

