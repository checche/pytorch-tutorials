import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_dataset():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = torchvision.datasets.FashionMNIST(
        './data',
        download=True,
        train=True,
        transform=transform
    )

    testset = torchvision.datasets.FashionMNIST(
        './data',
        download=True,
        train=False,
        transform=transform
    )

    return trainset, testset


def get_dataloader():
    trainset, testset = get_dataset()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    return trainloader, testloader


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)

    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
