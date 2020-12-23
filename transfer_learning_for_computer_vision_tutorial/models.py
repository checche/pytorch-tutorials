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


