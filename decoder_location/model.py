import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinedLocationDecoder(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1):
        super().__init__()
        
        # Initial reduction and resize to 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        )
        
        # Upsampling Stage 1: 64 -> 128
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling Stage 2: 128 -> 256
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling Stage 3: 256 -> 512
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Refinement
        self.refine = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: (B, 1024, 48, 48)
        
        x = self.conv1(x)
        
        x = self.up1(x)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = self.conv_up3(x)
        
        x = self.refine(x)
        
        x = self.final_conv(x)
        # x: (B, 1, 512, 512)
        
        return x
