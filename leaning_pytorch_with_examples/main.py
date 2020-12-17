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
