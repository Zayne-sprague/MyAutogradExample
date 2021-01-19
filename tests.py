from zensor import *
import torch
import numpy as np


def test_mult_singular():
    # Autograd code
    x = Zensor([4])
    x1 = Zensor([5])
    x2 = Zensor([10])

    y = x * (x1 * x) * x1 * x * x * x * x2

    y.backward()

    # Production code
    a = torch.tensor([4.], requires_grad=True)
    b = torch.tensor([5.], requires_grad=True)
    c = torch.tensor([10.], requires_grad=True)

    d = a * (b * a) * b * a * a * a * c
    d.retain_grad()
    d.backward()

    t_x = a.grad.detach().numpy()
    t_x1 = b.grad.detach().numpy()
    t_x2 = c.grad.detach().numpy()

    test = t_x == x.grad and t_x1 == x1.grad and t_x2 == x2.grad

    assert test, "Pytorch gradients for multiplication do not equal that of Zensors"


def test_mult_mat():
    def torch_v_zg(t, z):
        return (t.grad.detach().numpy() == z.grad).min()

    c_mult = 3
    A = np.array([[.1, .2, .3 ],] * c_mult, dtype=float).transpose()
    B = np.array([[.4, .5, .6],] * c_mult, dtype=float)
    C = np.array([[.7, .8, .9 ],], dtype=float).transpose()

    x = Zensor(A)
    x1 = Zensor(B)
    x2 = Zensor(C)

    t_x = torch.tensor(A, requires_grad=True)
    t_x1 = torch.tensor(B, requires_grad=True)
    t_x2 = torch.tensor(C, requires_grad=True)

    z = x @ x1
    y = zensor_Sigmoid(x2  - (z @ x2) ).exp().mean()
    output = y

    t_y = torch.sigmoid(t_x2 - (t_x @ t_x1 @ t_x2) ).exp().mean()
    t_y.retain_grad()
    t_y.backward()

    output.backward()

    grad_t_x = t_x.grad.detach().numpy()
    grad_t_x1 = t_x1.grad.detach().numpy()
    grad_t_x2 = t_x2.grad.detach().numpy()

    grad_x = x.grad
    grad_x1 = x1.grad
    grad_x2 = x2.grad




    test = False

    assert test, "Pytorch and Zensor mat mult gradient values do not match"


def test_mat_mult_operation():
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = [[3, 4, 3], [3, 5, 3], [3, 6, 3]]

    x = Zensor(A)
    x1 = Zensor(B)

    t_x = torch.tensor(A)
    t_x1 = torch.tensor(B)

    y = x * x1
    t_y = t_x * t_x1

    test = (t_y.numpy() == y.value).min()

    assert test, "Pytorch and Zensor mat mult not equal"


if __name__ == '__main__':
    # test_mult_singular()
    test_mult_mat()
    # test_mat_mult_operation()
