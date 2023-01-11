import torch.utils.data

from models.model.DeformRefine.deformable_refine import DeformableRefineF
from models.model.submodule import *


class LMNet(nn.Module):
    def __init__(self, maxdisp):
        super(LMNet, self).__init__()

        self.maxdisp = maxdisp
        self.mindisp = 0.001
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

        self.hourglass_size = 32
        self.pre_dres = convbn_3d(self.num_groups1 * 2, self.hourglass_size, 3, 1, 1)
        self.dres0_lea = nn.Sequential(ResidualBlock_3D(self.hourglass_size),
                                       ResidualBlock_3D(self.hourglass_size))
        self.dres1_lea = nn.Sequential(ResidualBlock_3D(self.hourglass_size),
                                       ResidualBlock_3D(self.hourglass_size))
        self.classif0_lea = nn.Sequential(
            convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
            nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.decoder1 = hourglass(self.hourglass_size)
        self.classif1 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
                                      nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.decoder2 = hourglass(self.hourglass_size)
        self.classif2 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
                                      nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.decoder3 = hourglass(self.hourglass_size)
        self.classif3 = nn.Sequential(convbn_3d(self.hourglass_size, self.hourglass_size, 3, 1, 1),
                                      nn.Conv3d(self.hourglass_size, 1, kernel_size=3, padding=1, stride=1, bias=False))

        disp_values = torch.arange(0, self.maxdisp, dtype=torch.float32)
        values_refine = disp_values.view(1, self.maxdisp, 1, 1)
        self.disp_values = nn.Parameter(values_refine, requires_grad=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
        basemodel_name1 = 'tf_efficientnet_b3_ap'
        basemodel1 = torch.hub.load('rwightman/gen-efficientnet-pytorch', basemodel_name1, pretrained=True)
        self.feature_extraction_lea = FeatureExtraction(basemodel=basemodel1)

    def forward(self, L, R):
        features_L_lea = self.feature_extraction_lea(L)
        features_R_lea = self.feature_extraction_lea(R)
        lea_cost_volume1 = self.cost_volume_1(features_L_lea[0], features_R_lea[0])
        lea_cost_volume2 = self.cost_volume_2(features_L_lea[1], features_R_lea[1])
        lea_cost_volume3 = self.cost_volume_3(features_L_lea[2], features_R_lea[2])
        lea_cost_volume = self.fuse1_lea(lea_cost_volume3, lea_cost_volume2)
        lea_cost_volume = self.fuse2_lea(lea_cost_volume, lea_cost_volume1)
        cost_lea = self.pre_dres(lea_cost_volume)
        if self.training:
            cost_lea = self.dres0_lea(cost_lea)
            cost_lea = self.dres1_lea(cost_lea) + cost_lea
            pred0_lea = self.classif0_lea(cost_lea)
            pred0_lea = F.interpolate(pred0_lea, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            pred0_lea = torch.squeeze(pred0_lea, 1)
            pred0_lea = F.softmax(pred0_lea, dim=1)
            pred0_lea = torch.sum(pred0_lea * self.disp_values, dim=1)

            cost1 = self.decoder1(cost_lea)
            pred1 = self.classif1(cost1)
            pred1 = F.interpolate(pred1, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            pred1 = torch.squeeze(pred1, 1)
            pred1 = F.softmax(pred1, dim=1)
            pred1 = torch.sum(pred1 * self.disp_values, dim=1)

            cost2 = self.decoder2(cost1)
            pred2 = self.classif2(cost2)
            pred2 = F.interpolate(pred2, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            pred2 = torch.squeeze(pred2, 1)
            pred2 = F.softmax(pred2, dim=1)
            pred2 = torch.sum(pred2 * self.disp_values, dim=1)

            cost3 = self.decoder3(cost2)
            cost3 = self.classif3(cost3)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = torch.sum(pred3 * self.disp_values, dim=1)
            return [pred0_lea.squeeze(1), pred1.squeeze(1), pred2.squeeze(1), pred3.squeeze(1)]
        else:
            cost_lea = self.dres0_lea(cost_lea)
            cost_lea = self.dres1_lea(cost_lea) + cost_lea

            cost1 = self.decoder1(cost_lea)
            cost2 = self.decoder2(cost1)
            cost3 = self.decoder3(cost2)
            cost3 = self.classif3(cost3)
            cost3 = F.interpolate(cost3, [self.maxdisp, L.size()[2], L.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = torch.sum(pred3 * self.disp_values, dim=1)
            return [pred3.squeeze(1)]


if __name__ == '__main__':
    model = LMNet(192).train()  # .cuda()
    #     state_dict = torch.load('../../ckpts/SCENEFLOW/gwc.ckpt', map_location='cpu')
    #     model.load_state_dict(state_dict['model'], strict=True)
    x1 = torch.rand([1, 3, 256, 512])  # .cuda()
    x2 = torch.rand([1, 3, 256, 512])  # .cuda()
    y = model(x1, x2)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
