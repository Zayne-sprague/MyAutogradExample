from autograd.visualization.GraphVis import *
from zensor.Zensor import Zensor
import torch

def gradcheck(func):
    def wrapped(*args, **kwargs):
        zensors, tensors = func(*args, **kwargs)

        for z, t in zip(zensors, tensors):
            z_grad = z.grad

            if not isinstance(z_grad, np.ndarray) and not z_grad:
                continue

            t_grad = t.grad.detach().numpy()

            check = np.all(t_grad == z_grad)
            if not check:
                return False

        return True

    return wrapped


def test_basic_zensor():
    # region Torch
    t_a = torch.tensor([10.], requires_grad=True)
    t_b = torch.tensor([20.], requires_grad=True)
    t_c = torch.tensor([5.], requires_grad=True)
    t_d = torch.tensor([7.], requires_grad=True)

    t_w = t_c * t_d + t_d + t_d
    t_x = t_w * t_w + t_a
    t_y = t_a + (t_b * t_b) + t_a + (t_b * t_a)
    t_z = t_y * t_y * t_a * (t_b + t_x)
    t_z = t_z + t_x

    t_a.retain_grad()
    t_b.retain_grad()
    t_c.retain_grad()
    t_d.retain_grad()
    t_w.retain_grad()
    t_x.retain_grad()
    t_y.retain_grad()
    t_z.retain_grad()

    # t_z.backward()
    t_z.backward()
    # endregion

    a = Zensor(10, symbol='a')
    b = Zensor(20, symbol='b')

    c = Zensor(5, symbol='c')
    d = Zensor(7, symbol='d')

    w = c * d + d + d
    w.symbol = 'w'
    x = w * w + a
    x.symbol = 'x'

    y = a + (b * b) + a + (b * a)
    y.symbol = 'y'

    z = y * y * a * ( b + x )
    z.symbol = 'z'

    z = z + x
    z.symbol = 'z'

    print(z.value)


    z.backward()
    print('hi')


def test_basic_backward():
    a = Zensor(10, symbol='a')
    b = Zensor(20, symbol='b')

    w = a * b
    w.symbol = 'w'

    # show(w.graph.build_nx())
    w.graph.backward()

    print(a.grad)

    show(a.grad.graph.build_nx())

    print(a)

def test_vector_math():
    #region Pytorch
    t_a = torch.tensor([1.,2.,3.,4.,5.,6.], requires_grad=True)
    t_b = torch.tensor([6.,5.,4.,3.,2.,1.], requires_grad=True)
    t_c = torch.tensor([3.,3.,3.,3.,3.,3.], requires_grad=True)

    t_y = t_a * t_b
    t_z = t_y @ t_c

    t_a.retain_grad()
    t_b.retain_grad()
    t_c.retain_grad()
    t_y.retain_grad()
    t_z.retain_grad()

    t_z.backward()

    #endregion

    a = Zensor([1, 2, 3, 4, 5, 6])
    b = Zensor([6, 5, 4, 3, 2, 1])
    c = Zensor([3, 3, 3, 3, 3, 3])

    y = a * b
    z = y @ c

    z.backward()

    print(z)


@gradcheck
def test_matrix_math():
    v1 = [1.,2.,3.,4.,5.,6.]
    v2 = [6.,5.,4.,3.,2.,1.]
    v3 = [3.,3.,3.]

    m1 = np.array([v1]*3)
    m2 = np.array([v2]*3)
    m3 = np.array([v3]*1)
    m4 = np.array([v3]*1)

    #region Pytorch
    t_a = torch.tensor(m1, requires_grad=True)
    t_b = torch.tensor(m2, requires_grad=True)
    t_c = torch.tensor(m3, requires_grad=True)
    t_d = torch.tensor(m4, requires_grad=True)

    t_y = t_a.transpose(0,1) @ t_b
    t_z = (t_d @ t_y[0:3,:])[:,0:3] @ t_c.transpose(0,1)
    t_z = t_z

    t_a.retain_grad()
    t_b.retain_grad()
    t_c.retain_grad()
    t_d.retain_grad()
    t_y.retain_grad()
    t_z.retain_grad()

    print(f'torch z: {t_z}')
    t_z.backward()

    #endregion

    a = Zensor(m1)
    b = Zensor(m2)
    c = Zensor(m3)
    d = Zensor(m4)

    y = a.transpose() @ b
    z = (d @ y[0:3,:])[:,0:3] @ c.transpose()
    z = z

    graph = build_computation_graph(z)
    show(graph)

    print(f'zensor z: {z}')
    z.backward()

    return (a, b, c, d, y, z), (t_a, t_b, t_c, t_d, t_y, t_z)

if __name__ == "__main__":
    # test_graph_creation()
    # test_basic_zensor()
    # test_basic_backward()
    # test_vector_math()
    test = test_matrix_math()
    print(test)