import math

import torch


def fit_autograd():
    dtype = torch.float
    device = torch.device('cpu')

    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    # requires_gradをTrueにすると
    # そのテンソルに関するloss関数の勾配計算をできるようになる。
    a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
    d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-6
    for t in range(2000):
        y_pred = a + b * x + c * x ** 2 + d * x ** 3

        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())

        # 自動微分をする。
        # requires_grad=Trueのテンソルに関するloss関数の勾配を計算する。
        # a.gradはaに関するloss関数の勾配を補持するテンソルになる。
        loss.backward()

        # 手動で重み更新するためにtorch.no_grad()でくるんでいる。
        # 自動微分で追跡する必要はありません
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            # 重み更新後に勾配を手動で0にする
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None

    print(
        f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')


class LegendrePolynominal3(torch.autograd.Function):
    """
    torch.autograd.Functionをサブクラス化し、
    ForwardとBackwardを実装することで、独自のautograd関数を実装できる。
    """

    @staticmethod
    def forward(ctx, input):
        """
        テンソルを入出力とする関数。
        ctxはBackward計算のためにスタッシュできるコンテキストオブジェクト。
        ctx.save_for_backendでBackwardパスで使うオブジェクトをキャッシュできる。
        """
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        出力に対する損失関数の勾配を含むテンソルを受け取り、
        入力に対する損失関数の勾配を計算する。
        """
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)


def fit_legendre():
    dtype = torch.float
    device = torch.device('cpu')

    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)

    a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
    c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 5e-6
    for t in range(2000):
        # Function.applyで宣言する。
        P3 = LegendrePolynominal3.apply

        y_pred = a + b * P3(c + d * x)

        loss = (y_pred - y).pow(2).sum()

        if t % 100 == 99:
            print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad

            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None
    print(
        f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')
