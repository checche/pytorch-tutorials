# %%
from IPython.display import display
import numpy as np
import torch


# %% [markdown]
# 宣言
# %%
display(
    torch.empty(5, 3),  # uninitialized
    torch.rand(5, 3),
    torch.zeros(5, 3, dtype=torch.long),
)

# %%
x = torch.tensor([5.5, 3])
x

# %%
# create a tensor based on an existing tensor.
x = x.new_ones(5, 3, dtype=torch.double)
display(x)
x = torch.randn_like(x, dtype=torch.float)
display(x)
# %%
x.size()  # this is in fact a tuple.

# %% [markdown]
# 演算
# %%
y = torch.rand(5, 3)
display(
    x + y,
    torch.add(x, y),
)
# %%
# providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
result
# %%
# in-place
# "_"が末尾にあるメソッドはin-place
y.add_(x)
y

# %% [markdown]
# その他操作

# %%
x[:, 1]

# %%
# reshape
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
display(x.size(), y.size(), z.size())
# %%
x = torch.randn(1)
display(x, x.item())
# %%
# torch tensor <-> numpy
a = torch.ones(5)
b = a.numpy()
display(a, b)

# メモリ共有しているのでどちらの値も変化する。
a.add_(1)
display(a, b)
# %%
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
display(a, b)
# %%
# cudaが使えるかどうか見たりできます
if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))

# %%
