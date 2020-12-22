import matplotlib.pyplot as plt
import numpy as np

import dataset
import models


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

