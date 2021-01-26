import math

import torch

import dataset


class ScratchLogSoftMax():
    def __init__(self):
        # initializing the weights here with Xavier initialisation
        # (by multiplying with 1/sqrt(n)).
        self.weights = torch.randn(784, 10) / math.sqrt(784)
        self.weights.requires_grad_()
        self.bias = torch.zeros(10, requires_grad=True)
        self.bs = 64

        (self.x_train,
         self.y_train,
         self.x_valid,
         self.y_valid) = dataset.make_dataset()
        self.n, self.c = self.x_train.shape

    def log_softmax(self, x):
        """"
        活性化関数
        値域は(-inf, 0)
        0に近づくほどソフトマックスの結果は1に近い
        """
        return x - x.exp().sum(-1).log().unsqueeze(-1)

    def model(self, xb):
        # @ はドット積の演算子
        return self.log_softmax(xb @ self.weights + self.bias)

    def nll(self, input, target):
        # 負の対数尤度関数 値域 [0, inf)
        # 各予測値から正解ラベルのlog_softmax値を取得し、
        # その平均を取ることで尤度をだす。
        # 完全に正確な予測ができていればnllは0になる
        # 平均値が大きいほど
        return -input[range(target.shape[0]), target].mean()

    def accuracy(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

    def train(self):
        lr = 0.5
        epochs = 2
        for epoch in range(epochs):
            for i in range((self.n - 1) // self.bs + 1):  # ミニバッチごとに処理
                start_i = i * self.bs
                end_i = start_i + self.bs
                xb = self.x_train[start_i:end_i]
                yb = self.y_train[start_i:end_i]
                pred = self.model(xb)
                loss_func = self.nll
                loss = loss_func(pred, yb)

                loss.backward()
                with torch.no_grad():
                    self.weights -= self.weights.grad * lr
                    self.bias -= self.bias.grad * lr
                    self.weights.grad.zero_()
                    self.bias.grad.zero_()
