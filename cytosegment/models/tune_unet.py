import torch
import torch.nn as nn
import torch.nn.functional as fun


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, relu=True):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int,
                1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # Select the activation function
        if relu:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(0.1, inplace=True)

        self.up = nn.Upsample(
            mode="bilinear", scale_factor=2, align_corners=True
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        # up_samp = self.up(psi)
        return x * psi


class EncodeBlock(nn.Module):
    """(CNN => BN => ReLU) * 2"""

    def __init__(
        self,
        in_size,
        out_size,
        conv_block,
        dilation,
        dropout,
        batch_norm,
        relu,
    ):
        super(EncodeBlock, self).__init__()

        # Create dilation kernel
        dilation_kernel = (dilation, dilation)
        # Select the activation function
        if relu:
            activation = nn.ReLU(inplace=True)
        else:
            activation = nn.LeakyReLU(0.1, inplace=True)

        # Create the block with a CNN layer
        block = [
            nn.Conv2d(
                in_size,
                out_size,
                kernel_size=(3, 3),
                padding=dilation,
                dilation=dilation_kernel,
            )
        ]
        # Add a batch normalization layer
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        # Add an activation layer
        block.append(activation)
        if conv_block == "double":
            # Add one more CNN layer if conv_block is 'double'
            block.append(
                nn.Conv2d(
                    out_size,
                    out_size,
                    kernel_size=(3, 3),
                    padding=dilation,
                    dilation=dilation_kernel,
                )
            )
            # Add a batch normalization layer
            if batch_norm:
                block.append(nn.BatchNorm2d(out_size))
            # Add an activation layer
            block.append(activation)

        # Add dropout layer
        if dropout > 0:
            block.append(nn.Dropout(p=dropout))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class DecodeBlock(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        up_mode,
        conv_block,
        dilation,
        dropout,
        batch_norm,
        relu,
        attention,
    ):
        super(DecodeBlock, self).__init__()
        self.attention = attention
        self.conv_block = EncodeBlock(
            in_size, out_size, conv_block, dilation, dropout, batch_norm, relu
        )

        self.attn_block = AttentionBlock(
            F_g=out_size, F_l=out_size, F_int=out_size // 2, relu=relu
        )

        if up_mode == "upconv":
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_size, out_size, kernel_size=(2, 2), stride=(2, 2)
                )
            )
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(
                    mode="bilinear", scale_factor=2, align_corners=True
                ),
                nn.Conv2d(in_size, out_size, kernel_size=(1, 1)),
            )

    def forward(self, prev_encode, curr_decode):
        up_prev_encode = self.up(prev_encode)
        if self.attention:
            atten_out = self.attn_block(up_prev_encode, curr_decode)
            cat_out = torch.cat((atten_out, up_prev_encode), dim=1)
            out = self.conv_block(cat_out)
            return out
        else:
            # Decode the final layer of encoding block
            # up_prev_encode = self.up(prev_encode)
            # Concat up and skip-connection layers
            x = torch.cat([curr_decode, up_prev_encode], dim=1)
            out = self.conv_block(x)
            return out


class TunableUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_classes=1,
        conv_block="double",
        depth=5,
        filters=6,
        dilation=1,
        dropout=0,
        up_mode="upconv",
        batch_norm=True,
        attention=False,
        relu=True,
    ):
        """
        Implementation of U-Net: Convolutional Networks for Biomedical
        Image Segmentation (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper

        Parameters
        ----------
            in_channels: int
                Number of input channels
            out_classes: int
                Number of output channels
            depth: int
                Depth of the network
            filters: int
                Number of filters in the CNN layers is created based on
                "filters" and "depth" arguments in a loop by using the
                below formula.
                for i in range(depth):
                    2**(filters + i)
            dilation: int
                Dilation rate to be introduced
            up_mode: str
                one of "upconv" or "upsample"."upconv" uses transposed
                convolutions for learned upsampling. "upsample" will use
                bilinear upsampling.
            with_attn: boolean
                specify whether attention blocks are included when building
                the model

        Returns
        -------
        """

        super(TunableUNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        assert conv_block in ("single", "double")

        prev_channels = in_channels

        # Create encoding path
        self.encoder = nn.ModuleList()
        for i in range(depth + 1):
            out_channels = 2 ** (filters + i)
            self.encoder.append(
                EncodeBlock(
                    prev_channels,
                    out_channels,
                    conv_block,
                    dilation,
                    dropout,
                    batch_norm,
                    relu,
                )
            )
            prev_channels = out_channels

        # Creating decoding path
        self.decoder = nn.ModuleList()
        for i in reversed(range(depth)):
            out_channels = 2 ** (filters + i)
            self.decoder.append(
                DecodeBlock(
                    prev_channels,
                    out_channels,
                    up_mode,
                    conv_block,
                    dilation,
                    dropout,
                    batch_norm,
                    relu,
                    attention,
                )
            )
            prev_channels = out_channels

        self.last = nn.Conv2d(prev_channels, out_classes, kernel_size=(1, 1))

    def forward(self, x):
        blocks = []
        for i, encode in enumerate(self.encoder):
            x = encode(x)
            if i != len(self.encoder) - 1:
                blocks.append(x)
                x = fun.max_pool2d(x, 2)

        for i, decode in enumerate(self.decoder):
            x = decode(x, blocks[-i - 1])
        logits = self.last(x)
        out = torch.sigmoid(logits)
        return out
