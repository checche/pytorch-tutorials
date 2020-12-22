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


def plot_classes_preds(net, images, labels):
    """(画像, 予測値, その確率, 正解ラベル, 正解かどうか)を出力"""
    preds, probs = models.images_to_probs(net, images)

    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title('{0}, {1:.1f}%\n(label: {2})'.format(
            dataset.classes[preds[idx]],
            probs[idx] * 100.0,
            dataset.classes[labels[idx]]),
            color=('green' if preds[idx] == labels[idx].item() else 'red')
        )
    return fig


def add_pr_curve_tensorboard(writer,
                             class_index,
                             test_probs,
                             test_preds,
                             global_step=0):
    """
    0から9までの「class_index」を取り込んで、
    それぞれの適合率-再現率曲線をプロットします
    """
    # 指定したclassのみTrueのboolean
    tensorboard_preds = test_preds == class_index
    # 全行から指定したclassの列を取得
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(dataset.classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()
