import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import dataset


def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class Lambda(nn.Module):
    """
    任意の関数オブジェクトを引数にすることで
    汎用的なforward passを生成できるモデル
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class WrappedDataLoader:
    """ジェネレータにpreprocessの処理を書けば前述のLambdaの処理を減らせる"""

    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class GenericImageCNN:
    def __init__(self):
        self.bs = 64
        self.epochs = 2
        self.lr = 0.1

    def get_model(self):
        # nn.Sequentialを使えばモデルを連結していくことでモデルを作成できる。
        model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 出力サイズを指定し、入力サイズは自由なモデル。
            Lambda(lambda x: x.view(x.size(0), -1)),
        )
        opt = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        return model, opt

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

    def fit(self, epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                self.loss_batch(model, loss_func, xb, yb, opt)

            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self.loss_batch(model, loss_func, xb, yb)
                      for xb, yb in valid_dl]
                )
                accuracies, nums_acc = zip(
                    *[self.accuracy(model(xb), yb)
                      for xb, yb in valid_dl]
                )

            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            accuracy = np.sum(np.multiply(
                accuracies, nums_acc)) / np.sum(nums_acc)

            print('epoch:', epoch, 'validloss:',
                  valid_loss, 'valid accuracy:', accuracy)

    def accuracy(self, out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean(), len(yb)

    def run(self):
        train_dl, valid_dl = dataset.get_data_loader(self.bs)

        train_dl = WrappedDataLoader(train_dl, preprocess)
        valid_dl = WrappedDataLoader(valid_dl, preprocess)

        model, opt = self.get_model()
        loss_func = F.cross_entropy
        self.fit(self.epochs, model, loss_func,
                 opt, train_dl, valid_dl)
