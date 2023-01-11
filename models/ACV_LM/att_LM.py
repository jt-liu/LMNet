from models.ACV_LM.cost_vol import *


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
                    y += 1
                    features.append(vi(features[-1]))
            else:
                if y == 8: break
                y += 1
                features.append(v(features[-1]))
        return [features[5], features[6], features[8]]


class LMNet(nn.Module):
    def __init__(self, maxdisp, outc):
        super(LMNet, self).__init__()

        self.maxdisp = maxdisp
        basemodel_name1 = 'tf_efficientnet_b3_ap'
        basemodel1 = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name1, pretrained=True)
        self.feature_extraction_lea = FeatureExtraction(basemodel=basemodel1)
        chann_fea1 = 32
        chann_fea2 = 48
        chann_fea3 = 136
        self.num_groups1 = chann_fea1 * 2 // 8  # 8
        self.num_groups2 = chann_fea2 * 2 // 8  # 12
        self.num_groups3 = chann_fea3 * 2 // 8  # 34

        self.cost_volume_1 = BuildCostVolume(chann_fea1 * 2, num_groups=self.num_groups1,
                                             volume_size=maxdisp // 4)
        self.cost_volume_2 = BuildCostVolume(chann_fea2 * 2, num_groups=self.num_groups2,
                                             volume_size=maxdisp // 8)
        self.cost_volume_3 = BuildCostVolume(chann_fea3 * 2, num_groups=self.num_groups3,
                                             volume_size=maxdisp // 16)
        self.fuse1_lea = FusionCostVolume(self.num_groups3, self.num_groups2)
        self.fuse2_lea = FusionCostVolume(self.num_groups2 * 2, self.num_groups1)
        self.out = conv3d_lrelu(self.num_groups1*2, outc, kernel_size=1, stride=1, pad=0)

    def forward(self, L, R):
        features_L_lea = self.feature_extraction_lea(L)
        features_R_lea = self.feature_extraction_lea(R)
        lea_cost_volume1 = self.cost_volume_1(features_L_lea[0], features_R_lea[0])
        lea_cost_volume2 = self.cost_volume_2(features_L_lea[1], features_R_lea[1])
        lea_cost_volume3 = self.cost_volume_3(features_L_lea[2], features_R_lea[2])
        lea_cost_volume = self.fuse1_lea(lea_cost_volume3, lea_cost_volume2)
        lea_cost_volume = self.fuse2_lea(lea_cost_volume, lea_cost_volume1)
        return self.out(lea_cost_volume)
