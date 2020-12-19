import math

from IPython.display import display
import numpy as np
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

    def loss_batch(self, model, loss_func, xb, yb, opt=None):
        """
        損失関数の計算処理をする関数
        trainingかvalidationかによってoptimizerを呼び出すか分岐している。
        """
        loss = loss_func(model(xb), yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb)

    def fit(self):
        model, opt = self.get_model()
        epochs = 2
        # TensorDatasetでデータセットを扱いやすくできる。
        train_ds = TensorDataset(self.x_train, self.y_train)
        # DataLoaderはバッチの管理を行う。自動的にミニバッチをイテレーションできる。
        train_dl = DataLoader(train_ds, batch_size=self.bs)

        valid_ds = TensorDataset(self.x_valid, self.y_valid)
        # 逆伝播をしないためメモリに余裕ができるので、バッチサイズを大きくした。
        # バッチ数を減らせば高速化できる。
        # バッチ数に関してメモリと速度にトレードオフがある。
        valid_dl = DataLoader(valid_ds, batch_size=self.bs * 2)
        for epoch in range(epochs):
            # これらのさまざまなフェーズで適切な動作を保証するために
            # 訓練前に呼び出す。
            model.train()
            for xb, yb in train_dl:
                self.loss_batch(model, self.loss_func, xb, yb, opt)

            # これらのさまざまなフェーズで適切な動作を保証するために
            # 推論前に呼び出す。
            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self.loss_batch(model, self.loss_func, xb, yb)
                      for xb, yb in valid_dl]
                )

            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            # epochごとにvalidation dataでlossの計算
            display(epoch, valid_loss)


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.randn(10))
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)
