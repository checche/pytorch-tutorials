import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision


import datasets


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                num_epochs=25):
    """
    1. 学習率のスケジューリング
    2. ベストモデルの保存
    をしながら学習する
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in datasets.dataloaders[phase]:
                inputs = inputs.to(datasets.device)
                labels = labels.to(datasets.device)

                optimizer.zero_grad()

                # 順伝播
                # set_grad_enabled()で追跡するかどうか決められる
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 訓練のときだけbackward と optimizeをする。
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            # データ1つあたりの損失
            epoch_loss = running_loss / datasets.dataset_sizes[phase]
            # 正答率
            epoch_acc = running_corrects.double(
            ) / datasets.dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # accuracyが最高記録になったときの重みをコピーする
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 各層のパラメータをコピー
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


def finetuning():
    """ファインチューニング"""
    model_ft = torchvision.models.resnet18(pretrained=True)

    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, len(datasets.class_names))

    model_ft = model_ft.to(datasets.device)

    criterion = nn.CrossEntropyLoss()

    # modelの全パラメータが最適化される
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # LR = learning_rate
    # 学習率を7エポックごとに0.1倍する
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=25)

    return model_ft


def fixed_feature_extractor():
    """特徴量抽出機"""
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv = model_conv.to(datasets.device)

    criterion = nn.CrossEntropyLoss()

    # 最終層(nn.Linear)だけが最適化される。
    optimizer_conv = optim.SGD(
        model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, num_epochs=25)

    return model_conv
