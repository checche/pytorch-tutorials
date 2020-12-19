# %%
from IPython.display import display

import dataset
import models

# Load the "autoreload" extension so that code can change
%load_ext autoreload

# always reload modules so that as you change code in src, it gets loaded
%autoreload 2

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
my_model.run()
# %%
