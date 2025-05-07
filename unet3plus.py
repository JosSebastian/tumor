import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super(Convolution, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if activation:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            Convolution(in_channels, out_channels),
            Convolution(out_channels, out_channels),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class Decoder(nn.Module):
    def __init__(self, skip_channels, bridge_channels=64):
        super(Decoder, self).__init__()

        self.adapters = nn.ModuleList()
        for in_channels in skip_channels:
            self.adapters.append(Convolution(in_channels, bridge_channels))

        self.fusion = Convolution(bridge_channels * len(skip_channels), bridge_channels)

    def forward(self, encoder_features, scale_factors):
        resized_features = []

        for feature, adapter, scale in zip(
            encoder_features, self.adapters, scale_factors
        ):
            if scale < 0:
                x = F.max_pool2d(feature, kernel_size=abs(scale), stride=abs(scale))
            elif scale > 0:
                x = F.interpolate(
                    feature, scale_factor=scale, mode="bilinear", align_corners=True
                )
            else:
                x = feature

            x = adapter(x)
            resized_features.append(x)

        x = torch.cat(resized_features, dim=1)
        x = self.fusion(x)

        return x


class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet3Plus, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        bridge_channels = 64

        # Encoder
        self.encoder1 = Encoder(in_channels, filters[0])
        self.encoder2 = Encoder(filters[0], filters[1])
        self.encoder3 = Encoder(filters[1], filters[2])
        self.encoder4 = Encoder(filters[2], filters[3])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            Convolution(filters[3], filters[4]), Convolution(filters[4], filters[4])
        )

        skip_channels = [
            [filters[0], filters[1], filters[2], filters[3], filters[4]],
            [filters[0], filters[1], filters[2], bridge_channels, filters[4]],
            [filters[0], filters[1], bridge_channels, bridge_channels, filters[4]],
            [filters[0], bridge_channels, bridge_channels, bridge_channels, filters[4]],
        ]

        # Decoder
        self.decoder4 = Decoder(skip_channels[0], bridge_channels)
        self.decoder3 = Decoder(skip_channels[1], bridge_channels)
        self.decoder2 = Decoder(skip_channels[2], bridge_channels)
        self.decoder1 = Decoder(skip_channels[3], bridge_channels)

        # Output
        self.output = nn.Conv2d(bridge_channels, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1_features, e1_pooled = self.encoder1(x)
        e2_features, e2_pooled = self.encoder2(e1_pooled)
        e3_features, e3_pooled = self.encoder3(e2_pooled)
        e4_features, e4_pooled = self.encoder4(e3_pooled)

        # Bottleneck
        bottleneck = self.bottleneck(e4_pooled)

        # Decoder
        d4 = self.decoder4(
            [e1_features, e2_features, e3_features, e4_features, bottleneck],
            [-8, -4, -2, 0, 2],
        )
        d3 = self.decoder3(
            [e1_features, e2_features, e3_features, d4, bottleneck],
            [-4, -2, 0, 2, 4],
        )
        d2 = self.decoder2(
            [e1_features, e2_features, d3, d4, bottleneck],
            [-2, 0, 2, 4, 8],
        )
        d1 = self.decoder1(
            [e1_features, d2, d3, d4, bottleneck],
            [0, 2, 4, 8, 16],
        )

        # Output
        o = self.output(d1)

        return o


class LateFusionUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(LateFusionUNet, self).__init__()

        self.unet1 = UNet3Plus(in_channels, out_channels)
        self.unet2 = UNet3Plus(in_channels, out_channels)

        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        out1 = self.unet1(x1)
        out2 = self.unet2(x2)

        x = torch.cat([out1, out2], dim=1)
        output = self.fusion(x)

        return output
