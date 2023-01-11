import torch
from torch import nn
import torch.nn.functional as F


def convbn_3d(in_channels, out_channels, kernel_size, stride, pad, groups=1, dilation=(1, 1, 1)):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad,
                                   groups=groups, dilation=dilation, bias=False),
                         nn.GroupNorm(out_channels // 4, out_channels),
                         nn.SiLU(True))


class ResidualConvUnit_3D(nn.Module):
    def __init__(self, features, receptive_field):
        super().__init__()
        self.receptive_field = receptive_field
        self.padding = (self.receptive_field - 1) // 2
        self.kernel_size = 3
        self.dilation = ((self.receptive_field - self.kernel_size) // (self.kernel_size - 1)) + 1
        self.conv = nn.Sequential(convbn_3d(features, features // 2, 1, 1, 0),
                                  convbn_3d(features // 2, features // 2, (1, self.kernel_size, self.kernel_size), 1,
                                            pad=(0, self.padding, self.padding),
                                            dilation=(1, self.dilation, self.dilation)),
                                  convbn_3d(features // 2, features, 1, 1, 0))

    def forward(self, x):
        out = self.conv(x)
        return out + x


'''class ResidualConvUnit_3D(nn.Module):
    def __init__(self, features):
        super().__init__()
        kernel_size = 31
        pad = 15
        self.conv1 = nn.Sequential(convbn_3d(features, features, 1, 1, 0),
                                   convbn_3d(features, features, (1, 1, kernel_size), 1, pad=(0, 0, pad),
                                             groups=features),
                                   convbn_3d(features, features, 1, 1, 0),
                                   convbn_3d(features, features, (1, kernel_size, 1), 1, pad=(0, pad, 0),
                                             groups=features),
                                   convbn_3d(features, features, 1, 1, 0))
        self.conv2 = nn.Sequential(convbn_3d(features, features, 1, 1, 0),
                                   convbn_3d(features, features, (1, kernel_size, 1), 1, pad=(0, 0, pad),
                                             groups=features),
                                   convbn_3d(features, features, 1, 1, 0),
                                   convbn_3d(features, features, (1, 1, kernel_size), 1, pad=(0, pad, 0),
                                             groups=features),
                                   convbn_3d(features, features, 1, 1, 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out + x)
        return out'''


class ResidualBlock_3D(nn.Module):
    def __init__(self, num_cost_volume_channels):
        super(ResidualBlock_3D, self).__init__()
        self.res1 = ResidualConvUnit_3D(num_cost_volume_channels, 3)
        self.res2 = ResidualConvUnit_3D(num_cost_volume_channels, 3)
        self.res3 = ResidualConvUnit_3D(num_cost_volume_channels, 3)
        self.res4 = ResidualConvUnit_3D(num_cost_volume_channels, 3)

    def forward(self, x):
        x_1 = self.res1(x)
        x_2 = self.res2(x_1)
        x_3 = self.res3(x_2 + x)
        x_4 = self.res4(x_3 + x_1)
        return x_4


class hourglass1(nn.Module):
    def __init__(self, in_channels):
        super(hourglass1, self).__init__()

        self.conv1 = convbn_3d(in_channels, in_channels, 3, 1, 1)
        self.down_conv_1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 1, 1, 0),
                                         nn.AvgPool3d(kernel_size=2, stride=2))
        self.conv2 = ResidualBlock_3D(in_channels * 2)
        self.up_conv_2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels, 1, 1, 0),
                                       nn.Upsample(scale_factor=2.0, mode='trilinear'))
        self.conv5 = convbn_3d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x_in = self.conv1(x)
        x_down_1 = self.down_conv_1(x_in)
        x_down_1 = self.conv2(x_down_1)
        x_up_2 = self.up_conv_2(x_down_1)
        x_up_2 = self.conv5(x_up_2 + x_in)
        return x_up_2


class hourglass2(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2, self).__init__()

        self.conv1 = ResidualBlock_3D(in_channels)
        self.down_conv_1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 1, 1, 0),
                                         nn.AvgPool3d(kernel_size=2, stride=2))
        self.conv2 = ResidualBlock_3D(in_channels * 2)
        self.down_conv_2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 1, 1, 0),
                                         nn.AvgPool3d(kernel_size=2, stride=2))
        self.conv3 = ResidualBlock_3D(in_channels * 4)
        self.up_conv_1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 1, 1, 0),
                                       nn.Upsample(scale_factor=2.0, mode='trilinear'))
        self.conv4 = ResidualBlock_3D(in_channels * 2)
        self.up_conv_2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels, 1, 1, 0),
                                       nn.Upsample(scale_factor=2.0, mode='trilinear'))
        self.conv5 = ResidualBlock_3D(in_channels)

    def forward(self, x):
        x_in = self.conv1(x)
        x_down_1 = self.down_conv_1(x_in)
        x_down_1 = self.conv2(x_down_1)
        x_down_2 = self.down_conv_2(x_down_1)
        x_down_2 = self.conv3(x_down_2)
        x_up_1 = self.up_conv_1(x_down_2)
        x_up_1 = self.conv4(x_up_1 + x_down_1)
        x_up_2 = self.up_conv_2(x_up_1)
        x_up_2 = self.conv5(x_up_2 + x_in)
        return x_up_2


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = convbn_3d(in_channels, in_channels * 2, 3, 2, 1)

        self.conv2 = convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1)

        self.conv3 = convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1)

        self.conv4 = convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.GroupNorm(in_channels // 2, in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.GroupNorm(in_channels // 4, in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6
