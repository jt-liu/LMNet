import torch
from torch import nn
import torch.nn.functional as F

###############################################################################
""" Loss Function """


###############################################################################

def pixel_loss(disp_est, disp_gt, mask):
    losses = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean')
    return losses


def error(disp_est, disp_gt, mask):
    losses = F.l1_loss(disp_est[mask], disp_gt[mask], reduction='mean')
    return losses


def model_loss(disp_ests, disp_gt, mask, epoch_idx, mode):
    if mode == 'train':
        pixel_0 = pixel_loss(disp_ests[0], disp_gt, mask) * 0.5
        pixel_1 = pixel_loss(disp_ests[1], disp_gt, mask) * 0.5
        pixel_2 = pixel_loss(disp_ests[2], disp_gt, mask) * 0.7
        pixel_3 = pixel_loss(disp_ests[3], disp_gt, mask) * 1.0
        loss = pixel_0 + pixel_1 + pixel_2 + pixel_3
        if loss.data.item() > 1: loss = torch.sqrt(loss)
        if torch.isnan(loss): loss.data = torch.Tensor([0.001]).cuda()
        return loss
    elif mode == 'test':
        loss1 = []
        for i in range(len(disp_ests)):
            loss1.append(error(disp_ests[i], disp_gt, mask))
        loss = sum(loss1)
        if torch.isnan(loss): loss.data = torch.Tensor([0.])
        return loss
