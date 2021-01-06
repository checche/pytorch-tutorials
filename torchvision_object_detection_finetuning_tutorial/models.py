import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def faster_rcnn_finetuning():
    # COCOで訓練済みのモデルを読み込む
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # 分類機をnum_classesを持つ分類機にする。
    num_classes = 2  # person + background

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def faster_rcnn_alt_backbone():
    # 特徴量のみを取得
    backbone = torchvision.models.mobilentet_v2(pretrained=True).features

    # FasterRCNNはバックボーンからの出力チャネル数を知る必要がある
    # mobilenet_v2の出力は1280
    backbone.out_channels = 1280

    # RPN: Resion Proposal Networkに空間ごとにアンカーを生成。
    # 5つのサイズと3つのアスペクト比が存在することを意味します。
    # 特徴マップごとに異なるサイズとアスペクト比となる可能性があるので，
    # Tuple[Tuple[int]] という形式で指定します。
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # 関心領域のトリミングを実行するために使用する特徴マップ(featmap_names)と
    # 画像の大きさをもとに戻したあとのトリミングのサイズ(output_size)を適宜しましょう。
    # バックボーンがテンソルを返す場合、featmap_nameは[0]になっているはずです。
    # もう少し一般化して説明すると、バックボーンはOrderedDict[Tensor]を返すことになるので、
    # featmap_namesで使用する特徴マップを選択できます。
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def get_instance_segmentation_model(num_classes):
    # 訓練済みインスタンスセグメンテーションのモデルを読み込む
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # セグメンテーションマスクの分類器に入力する特徴量を取得
    in_features_mask = model.roi_heads.mask_predicator.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
