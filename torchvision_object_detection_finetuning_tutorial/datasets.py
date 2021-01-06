import os

import numpy as np
from PIL import Image
import torch
import torch.utils.data


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(
            sorted(os.listdir(os.path.join(self.root, 'PNGImages'))))
        self.masks = list(
            sorted(os.listdir(os.path.join(self.root, 'PedMasks'))))

    def __getitem__(self, idx):
        # 画像の読み込み
        img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])
        img = Image.open(img_path).convert('RGB')

        # 各色はインスタンスを表現しているため、RGBに変換しない
        mask = Image.open(mask_path)

        # 物体を表す座標にそのIDが入っている
        mask = np.array(mask)

        # 物体のidが並んだarray
        obj_ids = np.unique(mask)

        # 0番目のIDは背景なので除く
        obj_ids = obj_ids[1:]

        # ブロードキャストを利用して
        # インスタンスごとに、その場所をTrueにした配列を作成
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []

        for i in range(num_objs):
            # trueの箇所のインデックスを返す
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 1がnum_objs個あるテンソル
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # すべてFalseと仮定している
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
