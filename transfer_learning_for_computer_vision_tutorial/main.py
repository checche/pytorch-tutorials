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

%load_ext autoreload  # noqa
%autoreload 2  # noqa

# %%[markdown]
# ## データの読み込み
# 今回のデータセットはアリとハチのそれぞれ訓練用約120枚、検証用に75枚画像が用意されている。
# しかしこれではスクラッチでに実装だとデータ数が不足している。
# 転移学習(ファインチューニング)を使用して効率的にモデルを汎化させられる。

# %%[markdown]
# ## visualize a few images
# %%
plots.im_batch_show()
# %%[markdown]
# ## Training the model
# models.train_modelを参照
# %%[markdown]
# ## Visualzing the model predictions
# plots.visualize_modelを参照
# %%[markdown]
# ## Finetuning the convnet
# %%
# CPUだと15-25分
# GPUだと1分以内で終わる。
model_ft = models.finetuning()
# %%
plots.visualize_model(model_ft)

# %% [markdown]
# Conv Netを特徴抽出機として使う
# 最後の全結合層を除く全てのネットワークのパラメータを固定する。
# 固定にはrequires_grad=Falseを使う。
# %%
model_conv = models.fixed_feature_extractor()

# %%
plots.visualize_model(model_conv)

# %%
