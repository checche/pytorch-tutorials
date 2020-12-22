import torch
import torchvision
import torchvision.transforms as transforms


classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


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


def select_n_random(data, labels, n=100):
    """
    ランダムにn個のデータ点とそのラベルを選択
    """
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
