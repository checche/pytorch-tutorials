# %% [markdown]
# # VISUALIZING MODELS, DATA, AND TRAINING WITH TENSORBOARD
# TensorBoardをつかってトレーニングの実行結果を視覚化できる。

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import dataset
import models
import plot

# Load the "autoreload" extension so that code can change
%load_ext autoreload

# always reload modules so that as you change code in src, it gets loaded
%autoreload 2
# %%
trainloader, testloader = dataset.get_dataloader()
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# %%
net = models.Net()


# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %% [markdown]
# ## TensorBoard setup
# デフォルトではrunsに保存される。
# 今回はruns/fashion_mnist_experiment_1
# %%
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
# %% [markdown]
# Writing to TensorBoard
# %%
dataiter = iter(trainloader)
images, labels = dataiter.next()

img_grid = torchvision.utils.make_grid(images)

plot.matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('four_fashion_mnist_images', img_grid)
# %% [markdown]
# ターミナルでこちらを実行`tensorboard --logdir=runs`
# %% [markdown]
# ## TensorBoardでモデルをよく見てみる
# %%
# グラフを描画できる
writer.add_graph(net, images)
writer.close()

# %% [markdown]
# ## Adding a "Projector" to TensorBoard
# add_embedding methodで方次元のデータを低次元にして可視化できる
# %%
trainset, testset = dataset.get_dataset()
# ランダムに画像とそのラベル番号を取得
images, labels = dataset.select_n_random(trainset.data, trainset.targets)

# 各画像のクラスラベルを取得
class_labels = [classes[lab] for lab in labels]

features = images.view(-1, 28 * 28)
writer.add_embedding(
    features,
    metadata=class_labels,
    label_img=images.unsqueeze(1)  # 指定した位置にサイズ1の新しい次元を追加したテンソルを返す
)

writer.close()
# %% [markdown]
# ## TensorBoardでモデルのトレーニングを追跡する。
# plot_classes_preds関数を介してモデルが行っている予測のビューとともに、
# 実行中の損失をTensorBoardに記録します

# %%
running_loss = 0.0
for epoch in range(1):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            # 1000回の平均ロスを追加
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(trainloader) + i)
            writer.add_figure('predictions vs. actuals',
                              plot.plot_classes_preds(net, inputs, labels),
                              global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Traing')

# %% [markdown]
# ## TensorBoardで訓練済みモデルの評価をする

# %%
# バッチごとのソフトマックスの結果の配列
class_probs = []
# バッチごとの予測結果の配列
class_preds = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

# 各テストデータのprobs(size: num_data, num_class), preds(size: num_data)の並べたもの
test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

# %%
for i in range(len(classes)):
    plot.add_pr_curve_tensorboard(writer, i, test_probs, test_preds)
