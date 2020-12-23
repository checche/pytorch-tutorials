# %% [markdown]
# # Transfer Learning for Computer Vision Tutorial
# ImageNetなどの学習済みデータセットを利用してConvNetを事前学習し、
# このConvNetを初期値もしくは特徴量抽出機として、タスクで活用する。
# - ConvNetをファインチューニングする。
#   訓練済みパラメータを訓練するネットワークの初期値として利用。
#   その後は通常通りネットワークを訓練します。
# - ConvNetを特徴量抽出器として使う。
#   最後の全結合層を除いて訓練済みネットワークの重みを固定する。
#   次に最後の全結合層のみをランダムな重みを持つ新たなものに置き換える。
#   最終層だけを訓練する。

# %%
import datasets
import models
import plots

%load_ext autoreload
%autoreload 2

# %%[markdown]
# ## データの読み込み
# 今回のデータセットはアリとハチのそれぞれ訓練用約120枚、検証用に75枚画像が用意されている。
# しかしこれではスクラッチでに実装だとデータ数が不足している。
# 転移学習(ファインチューニング)を使用して効率的にモデルを汎化させられる。

# %%[markdwon]
# ## visualize a few images
# %%
plots.im_batch_show()
