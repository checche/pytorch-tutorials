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
import datasets

# Load the "autoreload" extension so that code can change
%load_ext autoreload

# always reload modules so that as you change code in src, it gets loaded
%autoreload 2


# %%
dataset = datasets.PennFudanDataset('./data/PennFudanPed')
dataset[0]

