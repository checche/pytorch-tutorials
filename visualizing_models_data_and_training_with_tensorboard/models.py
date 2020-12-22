import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def images_to_probs(net, images):
    """ラベルの予測とその確率を出力(size: batch_size)
    F.softmax(el, dim=0) -> 出力(size:10)に
    ソフトマックスを適用して得られた確率(size:10)が得られる。
    """
    output = net(images)
    # 最大値とそのidxが得られる
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())  # squeeze: サイズ1に次元を取り除く(次元を絞る)
    return preds, [F.softmax(el, dim=0)[i].item()
                   for i, el in zip(preds, output)]
