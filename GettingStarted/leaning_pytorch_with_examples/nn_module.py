import math
import random

import torch


def fit_nn():
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    # (x, x^2, x^3)
    # x.unsqueeze(-1) -> (2000, 1)
    # xx -> (2000, 3)
    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    # Sequential: 引数のnnを順番に適用し、出力を生成
    # Linear: 線形関数を使用して出力を計算。重みとバイアスを内部に持つ
    # Flatten: 線形レイヤーの出力を1Dのテンソルにしてyに一致させる。
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )

    # 損失関数
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-6
    for t in range(2000):
        y_pred = model(xx)

        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # backwardの実行前に勾配を0にする。
        model.zero_grad()

        # モデルのすべての学習可能なパラメータに関するlossの勾配を計算する。
        # 内部的には、各モジュールのパラメータはrequires_grad=Trueでテンソルに格納されるため
        # この呼出によって全パラメータの勾配が計算される。
        loss.backward()

        # 勾配降下法で重み更新。パラメータはテンソルなのでその勾配にアクセス可能
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad

    # リストのような操作で、1層目にアクセスすることができる。
    linear_layer = model[0]

    # For linear layer, its parameters are stored as `weight` and `bias`.
    print(f'Result: y = {linear_layer.bias.item()}'
          f' + {linear_layer.weight[:, 0].item()} x'
          f' + {linear_layer.weight[:, 1].item()} x^2'
          f' + {linear_layer.weight[:, 2].item()} x^3'
          )


def fit_optim():
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    p = torch.tensor([1, 2, 3])
    xx = x.unsqueeze(-1).pow(p)

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 1),
        torch.nn.Flatten(0, 1)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # optimizerの引数には更新したいテンソルを書く
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    for t in range(2000):
        y_pred = model(xx)

        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        # backwardパスの前にモデルの学習可能な重みに関する勾配を0にする
        # .backward()が呼び出されるたびにバッファーに蓄積されるためである。
        # 勾配が上書きされることはない。蓄積される。
        optimizer.zero_grad()

        # 各パラメータに関して勾配計算
        loss.backward()

        # step()で重み更新できる。
        optimizer.step()

    linear_layer = model[0]

    print(f'Result: y = {linear_layer.bias.item()}'
          f' + {linear_layer.weight[:, 0].item()} x'
          f' + {linear_layer.weight[:, 1].item()} x^2'
          f' + {linear_layer.weight[:, 2].item()} x^3'
          )


class Polynomial3(torch.nn.Module):
    def __init__(self) -> None:
        """
        パラメータを定義/初期化
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        入力データのテンソルを受け入れ、出力データのテンソルを返す。
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        return (f'y = {self.a.item()}'
                f' + {self.b.item()} x'
                f' + {self.c.item()} x^2'
                f' + {self.d.item()} x^3')


def fit_custom_nn():
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    model = Polynomial3()

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    for t in range(2000):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Result: {model.string()}')


class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        4次または4次と5次の項をランダムに追加する。何も追加しない場合もある。
        forward passにはPythonの制御フローを使用できる。
        """
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        """
        Just like any class in Python,
        you can also define custom method on PyTorch modules
        """
        return (f'y = {self.a.item()}'
                f' + {self.b.item()} x'
                f' + {self.c.item()} x ^ 2'
                f' + {self.d.item()} x ^ 3'
                f' + {self.e.item()} x ^ 4 ?'
                f' + {self.e.item()} x ^ 5 ?'
                )


def fit_control_flow():
    x = torch.linspace(-math.pi, math.pi, 2000)
    y = torch.sin(x)

    model = DynamicNet()

    # 普通のSGDでは終わらないのでモメンタムを使う
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

    for t in range(30000):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if t % 2000 == 1999:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Result: {model.string()}')
