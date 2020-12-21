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

# Load the "autoreload" extension so that code can change
%load_ext autoreload

# always reload modules so that as you change code in src, it gets loaded
%autoreload 2

# Load the TensorBoard notebook extension
%load_ext tensorboard
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

dataset.matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('four_fashion_mnist_images', img_grid)
# %% [markdown]
# ターミナルでこちらを実行`tensorboard --logdir=runs`
# %% [markdown]
# ## TensorBoardでモデルをよく見てみる
# %%
# グラフを描画できる
writer.add_graph(net, images)
writer.close()
# %%
