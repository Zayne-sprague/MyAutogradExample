import torch
from zensor import Zensor

# region old stuff
def test(x):
    y = x * x * x
    return y

def wraps(fun, namestr="{fun}", docstr="{doc}", **kwargs):
    def _wraps(f):
        print("INSIDE _WRAPS")
        f.__name__ = f"{getattr(f, '__name__', '[unkown name]')} wrapped"
        return f

    return _wraps

def wrap_nary_f(fun, op, argnum):
    namestr = "{op}_of_{fun}_wrt_argnum_{argnum}"
    docstr = """\
    {op} of function {fun} with respect to argument number {argnum}. Takes the
    same arguments as {fun} but returns the {op}.
    """
    return wraps(fun, namestr, docstr, op="blah", argnum=argnum)

def unary_to_nary(unary_operator):
    # This first wrap, wraps the function with the decorator - i.e. grad, grad now becomes nary_operator that returns the actual grad func without the func arg.
    @wraps(unary_operator)
    def nary_operator(fun, argnum=0, *nary_op_args, **nary_op_kwargs):

        # This second wrap, wraps the function passed into the first
        @wrap_nary_f(fun, unary_operator, argnum)
        def nary_f(*args, **kwargs):

            # This third and final wrap, wraps the function passed into the wrapped func (grads input).
            @wraps(fun)
            def unary_f(x):
                return fun(x, **kwargs)
            x = args[argnum]
            return unary_operator(unary_f, x, *nary_op_args, **nary_op_kwargs)
        return nary_f
    return nary_operator

@unary_to_nary
def grad(func, x):
    return func(x)
#endregion



# grad_test = grad(test)

# my code
