import numpy as np


# Taken from Autograds unbroadcast
# Description: This function is a helper function making sure that the gradient output matches the structure of the
# the variable we are trying to calculate the gradient for.
def unbroadcast(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta

    while np.ndim(x) > target_ndim:
        x = np.sum(x, axis=broadcast_idx)

    for axis, size in enumerate(target_shape):
        if size == 1:
            x = np.sum(x, axis=axis, keepdims=True)

    if np.iscomplexobj(x) and not target_iscomplex:
        x = np.real(x)

    return x
