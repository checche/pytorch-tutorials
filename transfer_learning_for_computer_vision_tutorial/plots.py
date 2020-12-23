import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import datasets


def imshow(inp, title=None):
    """テンソルにImshowをする"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def im_batch_show():
    inputs, classes = next(iter(datasets.dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[datasets.class_names[x] for x in classes])


