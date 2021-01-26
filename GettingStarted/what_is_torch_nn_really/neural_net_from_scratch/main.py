# %%
from IPython.display import display

import dataset
import models

# %% [markdown]
# ## MNIST data setup
# %%
x_train, y_train, x_valid, y_valid = dataset.make_dataset()
# %%
n, c = x_train.shape
display(
    x_train,
    y_train,
    x_train.shape,
    y_train.min(),
    y_train.max()
)
# %% [markdown]
# ## Neural net from scratch (no torch.nn)
# %%
my_model = models.ScratchLogSoftMax()
# %%

bs = 64
xb = x_train[0:bs]
preds = my_model.model(xb)
display(preds[0], preds.shape)

# %%
loss_func = my_model.nll
yb = y_train[0:bs]
display(loss_func(preds, yb))

# %%
preds.shape
# %%
my_model.accuracy(preds, yb)
# %%
my_model.train()
# %%
out = my_model.model(x_valid)
display(
    my_model.nll(out, y_valid),
    my_model.accuracy(out, y_valid)
)
# %%
