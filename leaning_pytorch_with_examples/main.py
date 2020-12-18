# %%
import autograd
import nn_module
import tensors

# %% [markdown]
# ## Tensors
# ### numpyで微分などを実装
# %%
tensors.fit_np()

# %% [markdown]
# ### テンソルで実装
# テンソルの良いところ
# - GPU使える
# - 計算や勾配を追跡できる
# %%
tensors.fit_tensor()

# %% [markdown]
# ## Autograd
# グラフのノード　-> Tensor
# グラフのエッジ -> 入力テンソルから出力テンソルを生成する関数
# 自動微分によって微分を手計算で定義する必要がなくなる。
# %%
autograd.fit_autograd()

# %% [markdown]
# ### 新しい自動微分関数の定義
# autograd演算子は内部的には2つの関数でできている(forward, backward)
# - foward: 入力テンソルから出力テンソルを計算。
# - backward: あるスカラー値に関する出力テンソルの勾配を受け取り、
# 同じスカラー値に関する入力テンソルの勾配を計算する。
# %%
autograd.fit_legendre()

# %% [markdown]
# ## nn module
# ### nn
# nnパッケージはニューラルネットワークのモデル構築に使う。
# 損失関数の定義にも使う。
# nnの入出力はテンソル。nnは内部のパラメータを保持するテンソルを持つ。
# %%
nn_module.fit_nn()

# %% [markdown]
# ### optim
# torch.no_grad()でパラメータテンソルを手動更新していたが、しんどい。
# optimパッケージはいろいろ楽に実装できる。
# %%
nn_module.fit_optim()

# %% [markdown]
# ### custom nn modules
# nn.Moduleをサブクラス化して、forwardを定義することで
# カスタマイズできる。
# %%
nn_module.fit_custom_nn()

# %% [markdown]
# ### Control Flow + Weight Sharing
# forwardなどの中に普通にforループとかpythonの制御フローが使える。
# 同じパラメータを複数回利用すれば重み共有ができる。
# %%
nn_module.fit_control_flow()
