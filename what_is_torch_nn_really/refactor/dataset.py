# %%
import gzip
from pathlib import Path
import pickle

from IPython.display import display
import matplotlib.pyplot as plt
import requests
import torch


DATA_PATH = Path('data')
PATH = DATA_PATH / 'mnist'

PATH.mkdir(parents=True, exist_ok=True)

URL = 'https://github.com/pytorch/tutorials/raw/master/_static/'
FILENAME = 'mnist.pkl.gz'


def make_dataset():
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open('wb').write(content)

    with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid),
         _) = pickle.load(f, encoding='latin-1')

    plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
    display(x_train.shape)

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    return x_train, y_train, x_valid, y_valid
