import math

import torch
from torch import nn
import torch.nn.functional as F

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
        self.model = Mnist_Logistic()
        self.loss_func = F.cross_entropy

    def accuracy(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

    def fit(self):
        lr = 0.5
        epochs = 2
        for epoch in range(epochs):
            for i in range((self.n - 1) // self.bs + 1):  # ミニバッチごとに処理
                start_i = i * self.bs
                end_i = start_i + self.bs
                xb = self.x_train[start_i:end_i]
                yb = self.y_train[start_i:end_i]
                pred = self.model(xb)
                loss = self.loss_func(pred, yb)

                loss.backward()
                with torch.no_grad():
                    for p in self.model.parameters():
                        p -= p.grad * lr
                    self.model.zero_grad()


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.randn(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
