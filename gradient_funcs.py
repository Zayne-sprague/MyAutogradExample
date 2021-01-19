
import numpy as np

mult = lambda x : lambda g : x.transpose() * g

mat_mult_0 = lambda x : lambda g : np.matmul(x.transpose(), g)
mat_mult_1 = lambda x : lambda g : np.matmul(g, x.transpose())

add = lambda x : lambda g : g
vadd = lambda x : lambda g : g[0]

sum = lambda x, cloned : lambda g : np.array([g.tolist(),] * cloned).transpose()

sub_0 = lambda x : lambda g : g
sub_1 = lambda x : lambda g : -g

v_sub_0 = lambda x : lambda g : g[:,0]
v_sub_1 = lambda x : lambda g : -g[:,0]

div_0 = lambda x : lambda g: g / x
div_1 = lambda x, y : lambda g : - g * x / y ** 2

exp = lambda x, ans : lambda g : ans * g
log = lambda x, ans : lambda g : g / x

neg = lambda x : lambda g : -g

sigmoid = lambda x : lambda g : g * (np.exp(-x) / (np.square(1.0 + np.exp(-x))))