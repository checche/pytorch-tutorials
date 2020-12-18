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
# ##
# %%
my_model = models.ScratchLogSoftMax()

# %%
my_model.fit()
# %%
out = my_model.model(x_valid)
display(
    my_model.loss_func(out, y_valid),
    my_model.accuracy(out, y_valid)
)
# %%
