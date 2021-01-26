import math

import numpy as np
import torch


def fit_np():

    x = np.linspace(-math.pi, math.pi, 2000)
    y = np.sin(x)

    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()

    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        # ロスの計算
        loss = np.square(y_pred - y).sum()
        if t % 100 == 99:
            print(t, loss)

        # 勾配を求める
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # 重み更新
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')


def fit_tensor():
    dtype = torch.float
    device = torch.device('cpu')

    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    a = torch.randn((), device=device, dtype=dtype)
    b = torch.randn((), device=device, dtype=dtype)
    c = torch.randn((), device=device, dtype=dtype)
    d = torch.randn((), device=device, dtype=dtype)

    learning_rate = 1e-6
    for t in range(2000):
        # forward pass
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        loss = (y_pred - y).pow(2).sum().item()
        if t % 100 == 99:
            print(t, loss)

        # 勾配を求める
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * x).sum()
        grad_c = (grad_y_pred * x ** 2).sum()
        grad_d = (grad_y_pred * x ** 3).sum()

        # 重み更新
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d

    print(f'Result: y = {a.item()}'
          f' + {b.item()} x'
          f' + {c.item()} x^2'
          f' + {d.item()} x^3')
