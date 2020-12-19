import math

from IPython.display import display
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import dataset


class ScratchLogSoftMax():
    def __init__(self):
        self.bs = 64

        (self.x_train,
         self.y_train,
         self.x_valid,
         self.y_valid) = dataset.make_dataset()
        self.n, self.c = self.x_train.shape
        self.loss_func = F.cross_entropy
        self.lr = 0.5

    def accuracy(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

    def get_model(self):
        model = Mnist_Logistic()
        return model, optim.SGD(model.parameters(), lr=self.lr)

    def fit(self):
        model, opt = self.get_model()
        epochs = 2
        # TensorDatasetでデータセットを扱いやすくできる。
        train_ds = TensorDataset(self.x_train, self.y_train)
        # DataLoaderはバッチの管理を行う。自動的にミニバッチをイテレーションできる。
        train_dl = DataLoader(train_ds, batch_size=self.bs)
        for epoch in range(epochs):
            for xb, yb in train_dl:
                pred = model(xb)
                loss = self.loss_func(pred, yb)

                loss.backward()
                opt.step()
                opt.zero_grad()

        display(self.loss_func(model(xb), yb))


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.randn(10))
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)
