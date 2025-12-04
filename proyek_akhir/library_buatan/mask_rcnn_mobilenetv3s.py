import torch
import torchvision
from torchvision import models
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import os

def model_maskrcnnn():
    # ---------------------------------------
    # 1. Load backbone MobileNetV2 (pretrained)
    # ---------------------------------------
    inverted_residual_setting, last_channel = _mobilenet_v3_conf('mobilenet_v3_small')
    model1 = models.MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                        last_channel=last_channel,
                        num_classes=4)
    
    path = os.path.join( os.getcwd(), f"hasil/mobilenetv3small/apple_leaf_disease_20_mobilenetv3small_Adam_default_outchan1280_16batch/apple_leaf_disease_20_mobilenetv3small_Adam_default_outchan1280_16batch.pth")
    model1.load_state_dict(torch.load(path) )

    backbone = model1.features
    backbone.out_channels = 576

    # ---------------------------------------
    # 2. Anchor Generator (RPN)
    # ---------------------------------------
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),    # 5 ukuran anchor
        aspect_ratios=((0.5, 1.0, 2.0),)     # 3 rasio anchor
    )

    # ---------------------------------------
    # 3. ROI Poolers
    # ---------------------------------------
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],     # gunakan feature map index 0
        output_size=7,
        sampling_ratio=2
    )

    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=14,
        sampling_ratio=2
    )

    # ---------------------------------------
    # 4. Build Mask R-CNN Model
    # ---------------------------------------
    model = MaskRCNN(
        backbone=backbone,
        num_classes=5,                     # 4 objek + 1 background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
    )
    
    return model