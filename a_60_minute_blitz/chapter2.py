# %%
from IPython.display import display
import torch


# %% [markdown]
# - `x.requires_grad`: tensorの属性。これをTrueにすると操作の追跡開始
# - `out.backward()`: outの勾配計算
# - `x.grad`: outをxについての偏微分したときの勾配がここに累積される。
# - `x.detach()`: 追跡を解除
# - `x.no_grad()`: 追跡をしないようにするラッパー

# %% [markdown]
# - `.grad_fn`: テンソルを作成した関数を参照する

# %%
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
display(x, y, y.grad_fn)
# %%
z = y * y * 3
out = z.mean()
display(z, out)
# %%
a = torch.randn(2, 2)
a = ((a * 3) / (a-1))

display(a.requires_grad)
a.requires_grad_(True)
display(a.requires_grad)
b = (a * a).sum()
display(b.grad_fn)

# %% [markdown]
# ## 勾配

# %%
out.backward()  # outはスカラー
display(x.grad)  # xで偏微分

# %%
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
y
# %%
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)  # yはテンソル
x.grad

# %% [markdown]
# ## 追跡をやめる
# %%
display(
    x.requires_grad,
    (x ** 2).requires_grad
)
with torch.no_grad():
    display((x ** 2).requires_grad)
# %%
display(x.requires_grad)
y = x.detach()
display(
    y.requires_grad,
    x.eq(y).all(),
)
