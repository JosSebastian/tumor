import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolution, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convolution(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder_1 = Convolution(in_channels, 32)
        self.encoder_2 = Convolution(32, 64)
        self.encoder_3 = Convolution(64, 128)
        self.encoder_4 = Convolution(128, 256)
        self.encoder_5 = Convolution(256, 512)
        self.down = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = Convolution(512, 1024)

        # Decoder
        self.up_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_1 = Convolution(1024, 512)
        self.up_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_2 = Convolution(512, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_3 = Convolution(256, 128)
        self.up_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_4 = Convolution(128, 64)
        self.up_5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_5 = Convolution(64, 32)

        # Output
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(self.down(e1))
        e3 = self.encoder_3(self.down(e2))
        e4 = self.encoder_4(self.down(e3))
        e5 = self.encoder_5(self.down(e4))

        # Bottleneck
        bottleneck = self.bottleneck(self.down(e5))

        # Decoder
        d1 = self.up_1(bottleneck)
        d1 = torch.cat([e5, d1], dim=1)
        d1 = self.decoder_1(d1)

        d2 = self.up_2(d1)
        d2 = torch.cat([e4, d2], dim=1)
        d2 = self.decoder_2(d2)

        d3 = self.up_3(d2)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.decoder_3(d3)

        d4 = self.up_4(d3)
        d4 = torch.cat([e2, d4], dim=1)
        d4 = self.decoder_4(d4)

        d5 = self.up_5(d4)
        d5 = torch.cat([e1, d5], dim=1)
        d5 = self.decoder_5(d5)

        # Output
        o = self.output(d5)

        return o


class LateFusionUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(LateFusionUNet, self).__init__()

        self.unet1 = UNet(in_channels, out_channels)
        self.unet2 = UNet(in_channels, out_channels)

        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        out1 = self.unet1(x1)
        out2 = self.unet2(x2)

        x = torch.cat([out1, out2], dim=1)
        output = self.fusion(x)

        return output
