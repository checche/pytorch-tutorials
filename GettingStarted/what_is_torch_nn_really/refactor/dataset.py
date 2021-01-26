# %%
import gzip
from pathlib import Path
import pickle

import requests
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


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

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    return x_train, y_train, x_valid, y_valid


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs*2)
    )


def get_data_loader(bs):
    """訓練/検証のDataLoaderを返す"""
    x_train, y_train, x_valid, y_valid = make_dataset()
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)

    return train_dl, valid_dl
