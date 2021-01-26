# %% [markdown]
# # TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL
# インスタンスセグメンテーション: 同じ種類の物体を区別する
#
# セマンティックセグメンテーション: 同じ種類の物体を区別しない
# %% [markdown]
# ## Defining the Dataset
# https://github.com/pytorch/vision/tree/v0.3.0/references/detection
# これを参考にカスタムDatasetsを作成するとよい。
# torch.utils.data.Datasetを継承し、`__len__`と`__getitem__`を実装する。
# 今回`__getitem__`は以下のものを返せば良い
# - image: a PIL image of size (H, W)
# - target: a dict containing the following fields
#   - boxes (FloatTensor\[N, 4\]): N個の\[x0, y0, x1, y1\]形式のバウンディングボックス
#   - labels (Int64Tensor\[N\]): バウンディングボックスのクラスラベル。0は背景
#   - image_id(Int64Tensor\[1\]): 画像のID
#   - area (Tensor\[N\]): バウンディングボックスの面積。COCOでの評価時に
#     small, medium largeに分けるために使われる。
#   - iscrowd (UInt8Tensor\[N\]): if iscrowd=True, 評価時に無視される
#   - (optionally) masks (UInt8Tensor\[N, H, W\]): 各オブジェクトに対するセグメンテーションマスク
#   - (optionally) keypoints (FloatTensor\[N, K, 3\]): N個のそれぞれのオブジェクトに
#     対して、\[x, y, visibility\]形式のK個のキーポイントでオブジェクトを定義する。
#     visibility=0はキーポイントが画像上では見えないことを意味する。
#     データ拡張の際には、キーポイントの反転はデータの表現方法によっては不適切になることに注意。
#     その際に新しいキーポイントを表現するために`references/detection/transforms.py`をたぶん使うべき。

# %% [markdown]
# 注意点2つ
#
# 1. 背景が無い場合はもちろんlabelsには0を含まない。
# 2. トレーニング時にアスペクト比でグルーピングしたい場合、`get_height_and_width`メソッドを実装するとよい。
#   そうしないとすべての画像を読み込んでしまうので遅くなる。

# %%
from PIL import Image
import torch

import datasets
from engine import train_one_epoch, evaluate
import models
import utils

# Load the "autoreload" extension so that code can change
%load_ext autoreload

# always reload modules so that as you change code in src, it gets loaded
%autoreload 2


# %%
dataset = datasets.PennFudanDataset('./data/PennFudanPed')
dataset[0]

# %% [markdown]
# ## Defining your model
# Faster R-CNNは画像内の物体にたいして
# - バウンディングボックスの予測
# - オブジェクトのクラス予測
#
#
# をおこなう
# Mask R-CNNはFaster R-CNNに加えて各インスタンスの
# セグメンテーションマスクも予測する。
#
# torchvisionのmodelzooで使用可能なモデルは、以下の2ケースでよく利用される。
# 1. 訓練済みモデルの最終レイヤーをファインチューニングする。
# 2. モデルのメイン部分(Backbone)を別のモデルに置き換えるとき。
#
# models.pyを参照
# ### An Instance segmentation model for PennFudan Dataset
# データが少ないのでファインチューニングをする。
# Mask R-CNNを使用する。

# %% [markdown]
# ## Training and evaluation functions
# https://github.com/pytorch/vision/tree/v0.3.0/references/detection
# ここにいろいろなヘルパーがある。

# %%
dataset = datasets.PennFudanDataset(
    './data/PennFudanPed', datasets.get_transform(train=True))
dataset_test = datasets.PennFudanDataset(
    './data/PennFudanPed', datasets.get_transform(train=False))

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn
)

# %%
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2

model = models.get_instance_segmentation_model(num_classes)

model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# %%
num_epochs = 10

for epoch in range(num_epochs):
    # print_freq: 1エポック中のprintするまでのイテレーション数
    # cudaじゃないと動かないっぽい
    train_one_epoch(model, optimizer, data_loader,
                    device, epoch, print_freq=10)

    lr_scheduler.step()

    evaluate(model, data_loader_test, device=device)

# %%
# 予測してみる
img, _ = dataset_test[0]
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

prediction
# %%
# permute(): 次元の順序を入れ替える
Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

# %%
Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())

# %%
print(len(prediction[0]['masks']))
print(prediction[0]['scores'])
# %%
Image.fromarray(prediction[0]['masks'][1, 0].mul(255).byte().cpu().numpy())
