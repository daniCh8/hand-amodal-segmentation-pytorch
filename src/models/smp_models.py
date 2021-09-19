import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def USeResNext(num_channels, num_classes):
    return smp.Unet(encoder_name="se_resnext101_32x4d",
                    encoder_weights="imagenet",
                    in_channels=num_channels,
                    classes=num_classes)

def UPlusRegNet(num_channels, num_classes):
    return smp.UnetPlusPlus(encoder_name="timm-regnety_032",
                    encoder_weights="imagenet",
                    in_channels=num_channels,
                    classes=num_classes)

def DeepLabV3PlusB5(num_channels, num_classes):
    return smp.DeepLabV3Plus(encoder_name="efficientnet-b5",
                            encoder_weights="imagenet",
                            in_channels=num_channels,
                            classes=num_classes
                            )

def DeepLab_ResNext(num_channels, num_classes):  
    return smp.DeepLabV3Plus(encoder_name="se_resnext50_32x4d",
                            encoder_weights="imagenet",
                            in_channels=num_channels,
                            classes=num_classes
                            )

def PSPNetL2(num_channels, num_classes):
    return smp.PSPNet(encoder_name="timm-efficientnet-l2",
                    encoder_weights="noisy-student",
                    in_channels=num_channels,
                    classes=num_classes
                    )

def SPNResNet34(num_channels, num_classes):
    return smp.PAN(encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=num_channels,
                classes=num_classes
                )

def UXceptionNet(num_channels, num_classes):
    return smp.Unet(
                    encoder_name="xception",
                    encoder_weights="imagenet",
                    in_channels=num_channels,
                    classes=num_classes
                    )

def TryNet(num_channels, num_classes):
    return smp.PAN(
                    encoder_name="mobilenet_v2",
                    encoder_weights="imagenet",
                    in_channels=num_channels,
                    classes=num_classes
                    )
