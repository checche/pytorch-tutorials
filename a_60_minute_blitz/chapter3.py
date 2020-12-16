# %%
from IPython.display import display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# %% [markdown]
# ## ニューラルネットワークの定義
# %%


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1入力のイメージチャネル
        # 6出力のチャネル
        # 3 x 3の正方形による畳込みカーネル
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # アフィン演算: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling (2,2)の窓で
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 正方形の場合1つの数字だけ指定できる
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x) -> int:
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net
# %%
# 学習するパラメータを表示
params = list(net.parameters())
display(len(params), params[0].size())

# %%
input = torch.randn(1, 1, 32, 32)
out = net(input)
out

# %%
net.zero_grad()  # 勾配が累積されるため、勾配バッファを手動で0にする。
out.backward(torch.randn(1, 10))

# %% [markdown]
# ## Loss Function

# %%
output = net(input)
target = torch.randn(10)  # (1,)
target = target.view(1, -1)  # (1, 10)
criterion = nn.MSELoss()  # 平均二乗誤差

loss = criterion(output, target)
loss

# %% [markdown]
# 演算の流れはこんな感じになっている
# ```
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
# -> view -> linear -> relu -> linear -> relu -> linear
# -> MSELoss
# -> loss
# ```

# %%
display(
    loss.grad_fn,
    loss.grad_fn.next_functions[0][0],
    loss.grad_fn.next_functions[0][0].next_functions[0][0],
)

# %% [markdown]
# ## 逆伝播

# %%
net.zero_grad()  # zeroes the gradient buffers of all parameters
display(
    'conv1.bias.grad before backward',
    net.conv1.bias.grad
)

loss.backward()

display(
    'conv1.bias.grad after backward',
    net.conv1.bias.grad
)

# %% [markdown]
# ## 重みの更新

# %%
# 普通の勾配降下法
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# %%
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # 勾配バッファを0にする
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 重みの更新
