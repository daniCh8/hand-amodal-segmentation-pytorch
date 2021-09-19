import torch
import torch.nn as nn
import resnet_parts as parts

class UResNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1

        self.inc = parts.DoubleConv(n_channels, 32)
        
        self.conv1 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        self.layer1 = self._make_layer(parts.ResNetBasicBlock, 64, 64, 3)
        self.layer2 = self._make_layer(parts.ResNetBasicBlock, 64, 128, 4, stride=2)
        self.layer3 = self._make_layer(parts.ResNetBasicBlock, 128, 256, 6, stride=2)
        self.layer4 = self._make_layer(parts.ResNetBasicBlock, 256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.up1 = parts.Up(512 + 256, 512 // factor, bilinear)
        self.up2 = parts.Up(256 + 128, 256 // factor, bilinear)
        self.up3 = parts.Up(128 + 64, 128 // factor, bilinear)
        self.up4 = parts.Up(64 + 64, 32, bilinear)
        self.up5 = parts.Up(32 + 32, 16, bilinear)

        self.outc = parts.OutConv(16, n_classes)


    def _make_layer(self, block, in_filters, out_filters, blocks, stride=1):
        downsample = None  
        if stride != 1 or in_filters != out_filters:
            downsample = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 1, stride, bias=False),
                nn.BatchNorm2d(out_filters),
            )
        layers = []
        layers.append(block(in_filters, out_filters, stride, downsample=downsample)) 
        in_filters = out_filters
        for _ in range(1, blocks):
            layers.append(block(in_filters, out_filters))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # [3, 128, 128]
        x0 = self.inc(x)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu(x1) # [64, 64, 64]

        x2 = self.maxpool(x1) # [64, 32, 32]
        x2 = self.layer1(x2) # [64, 32, 32]

        x3 = self.layer2(x2) # [128, 16, 16]

        x4 = self.layer3(x3) # [256, 8, 8]

        x5 = self.layer4(x4) # [512, 4, 4] 
        x5 = self.avgpool(x5) #[512, 1, 1] 

        x = self.up1(x5, x4) # [256, 8, 8]
        x = self.up2(x, x3) # [128, 16, 16]
        x = self.up3(x, x2) # [64, 32, 32]
        x = self.up4(x, x1) # [64, 64, 64]
        x = self.up5(x, x0)

        logits = self.outc(x)
        return logits
