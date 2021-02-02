import uuid

"""
A Class Meant to describe various functions that could be used to create the Computation Graph
These include Multiplication/Addition/Exponentiation etc.
"""


class Function:

    def __init__(self, name):
        from zensor import Zensor

        self.name = name
        self.id = uuid.uuid4()

        self.__creator__ = Zensor

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def gradient_fn(self, ins, outs, grad):
        raise NotImplementedError()
