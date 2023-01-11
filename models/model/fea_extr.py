import torch
from torch import nn
import math


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.GroupNorm(out_channels // 2, out_channels))



class FeatureExtraction(nn.Module):
    def __init__(self, basemodel):
        super(FeatureExtraction, self).__init__()
        print('Building feature extraction model..', end='')
        print('Removing last layers.')
        basemodel.blocks[5] = nn.Identity()
        basemodel.blocks[6] = nn.Identity()
        basemodel.conv_head = nn.Identity()
        basemodel.bn2 = nn.Identity()
        basemodel.act2 = nn.Identity()
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel
        print('Done.')

    def forward(self, x):
        features = [x]
        y = 0
        for k, v in self.original_model._modules.items():
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    if y == 8: break
                    # print(y, k)
                    y += 1
                    features.append(vi(features[-1]))
            else:
                if y == 8: break
                # print(y, k)
                y += 1
                features.append(v(features[-1]))
        # for i in range(len(features)):
        #     print(i, features[i].shape)
        return [features[5], features[6], features[8]]
        # 24, 40, 112, 320 # 32, 48, 136, 384 # 40, 64, 176 # 32, 24, 40, 112

