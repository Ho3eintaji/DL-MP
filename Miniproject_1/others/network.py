import torch
import torch.nn as nn
from collections import OrderedDict


class UNet(nn.Module):

    def __init__(self, n_channels=3, init_features=48):
        super(UNet, self).__init__()

        block = UNet._block
        features = init_features

        # Encoder side
        self.encoder0 = block(n_channels, features, name="enc0")
        self.encoder1 = block(features, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = block(features, features, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = block(features, features, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = block(features, features, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = block(features, features, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder6 = block(features, features, name="enc6")

        # Decoder side
        self.upscale1 = nn.Upsample(scale_factor=2)
        self.decoder5 = block(features * 2, features * 2, name="dec5")
        self.decoder5b = block(features * 2, features * 2, name="dec5b")
        self.upscale2 = nn.Upsample(scale_factor=2)
        self.decoder4 = block(features * 3, features * 2, name="dec4")
        self.decoder4b = block(features * 2, features * 2, name="dec4b")
        self.upscale3 = nn.Upsample(scale_factor=2)
        self.decoder3 = block(features * 3, features * 2, name="dec3")
        self.decoder3b = block(features * 2, features * 2, name="dec3b")
        self.upscale4 = nn.Upsample(scale_factor=2)
        self.decoder2 = block(features * 3, features * 2, name="dec2")
        self.decoder2b = block(features * 2, features * 2, name="dec2b")
        self.upscale5 = nn.Upsample(scale_factor=2)
        self.decoder1 = block(features * 2 + n_channels, 64, name="dec1")
        self.decoder1b = block(64, 32, name="dec1b")

        # Output
        self.conv = nn.Conv2d(in_channels=32, out_channels=n_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Encode input
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc1 = self.pool1(enc1)
        enc2 = self.encoder2(enc1)
        enc2 = self.pool2(enc2)
        enc3 = self.encoder3(enc2)
        enc3 = self.pool3(enc3)
        enc4 = self.encoder4(enc3)
        enc4 = self.pool4(enc4)
        enc5 = self.encoder5(enc4)
        enc5 = self.pool5(enc5)
        enc6 = self.encoder6(enc5)

        # Decode using skip connections
        dec5 = self.upscale1(enc6)
        dec5 = torch.cat((dec5, enc4), dim=1)
        dec5 = self.decoder5b(self.decoder5(dec5))
        dec4 = self.upscale2(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.decoder4b(self.decoder4(dec4))
        dec3 = self.upscale3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.decoder3b(self.decoder3(dec3))
        dec2 = self.upscale4(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2b(self.decoder2(dec2))
        dec1 = self.upscale5(dec2)
        dec1 = torch.cat((dec1, x), dim=1)
        dec1 = self.decoder1b(self.decoder1(dec1))

        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name, **kwargs):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                ]
            )
        )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()
        self.blocks = ResBlock._block(in_channels, out_channels, name)
        self.activation = nn.ReLU()
        # If the output of block has different size from the input, pass the input through a convolution layer to
        # have both of them the same size. Otherwise, pass the input directly
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=3,
                                      padding=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Add the shortcut of the input to the output of blocks.
        x = self.shortcut(x) + self.blocks(x)
        x = self.activation(x)
        return x

    @staticmethod
    def _block(in_channels, features, name, **kwargs):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.LeakyReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )


class AE(nn.Module):

    def __init__(self, n_channels=3, init_features=48):
        super(AE, self).__init__()

        block = ResBlock
        features = init_features

        # Encoder side
        self.encoder0 = block(n_channels, features, name="enc0")
        self.encoder1 = block(features, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = block(features, features, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = block(features, features, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = block(features, features, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder5 = block(features, features, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder6 = block(features, features, name="enc6")

        # Decoder side
        self.upscale1 = nn.Upsample(scale_factor=2)
        self.decoder5 = block(features, features, name="dec5")
        self.upscale2 = nn.Upsample(scale_factor=2)
        self.decoder4 = block(features, features, name="dec4")
        self.upscale3 = nn.Upsample(scale_factor=2)
        self.decoder3 = block(features, features, name="dec3")
        self.upscale4 = nn.Upsample(scale_factor=2)
        self.decoder2 = block(features, features, name="dec2")
        self.upscale5 = nn.Upsample(scale_factor=2)
        self.decoder1 = block(features, features, name="dec1")

        # Output
        self.conv = nn.Conv2d(in_channels=features, out_channels=n_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Encode input
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc1 = self.pool1(enc1)
        enc2 = self.encoder2(enc1)
        enc2 = self.pool2(enc2)
        enc3 = self.encoder3(enc2)
        enc3 = self.pool3(enc3)
        enc4 = self.encoder4(enc3)
        enc4 = self.pool4(enc4)
        enc5 = self.encoder5(enc4)
        enc5 = self.pool5(enc5)
        enc6 = self.encoder6(enc5)

        # Decode the output of encoder
        dec5 = self.upscale1(enc6)
        dec5 = self.decoder5(dec5)
        dec4 = self.upscale2(dec5)
        dec4 = self.decoder4(dec4)
        dec3 = self.upscale3(dec4)
        dec3 = self.decoder3(dec3)
        dec2 = self.upscale4(dec3)
        dec2 = self.decoder2(dec2)
        dec1 = self.upscale5(dec2)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)


if __name__ == '__main__':
    model = AE()
    out = model(torch.zeros(1, 3, 32, 32))
    print(out.shape)
