import torch
import torch.nn as nn
import models.nnet_parts as parts

class UXNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UXNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = parts.DoubleConv(n_channels, 32)

        self.conv1 = nn.Conv2d(32,32,3,2,1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,64,3,padding=1,bias=False) # (64,64,64)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1=parts.Block(64,128,2,2,start_with_relu=False,grow_first=True) # (128, 32, 32)
        self.bn3 = nn.BatchNorm2d(128)
        self.block2=parts.Block(128,256,2,2,start_with_relu=True,grow_first=True) # (256, 16, 16)
        self.bn4 = nn.BatchNorm2d(256)
        self.block3=parts.Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=parts.Block(728,728,3,1,start_with_relu=True,grow_first=True) # (728, 8, 8)
        self.bn5 = nn.BatchNorm2d(728)
        
        factor = 2 if bilinear else 1
        
        self.up1 = parts.Up(728+256, 728 // factor, bilinear)
        self.up2 = parts.Up(364+128, 256 // factor, bilinear)
        self.up3 = parts.Up(128+64, 128 // factor, bilinear)
        self.up4 = parts.Up(64+32, 64, bilinear)
        self.outc = parts.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x) # (32,128,128)

        x2 = self.conv1(x1)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        
        x2 = self.conv2(x2) # (64,64,64)
        x2 = self.bn2(x2)
        x2 = self.relu(x2) 
        
        x3 = self.block1(x2) # (128, 32, 32)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        x4 = self.block2(x3) # (256, 16, 16)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)

        x5 = self.block3(x4)
        x5 = self.block4(x5)
        x5 = self.block5(x5)
        x5 = self.block6(x5)
        x5 = self.block7(x5)
        x5 = self.block8(x5)
        x5 = self.block9(x5)
        x5 = self.block10(x5)
        x5 = self.block11(x5) 
        x5 = self.bn5(x5)
        x5 = self.relu(x5) # (728, 8, 8)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits