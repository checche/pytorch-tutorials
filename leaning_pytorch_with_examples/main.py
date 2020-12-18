# %%
import autograd
import tensors

# Load the "autoreload" extension so that code can change
%load_ext autoreload

# always reload modules so that as you change code in src, it gets loaded
%autoreload 2

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
