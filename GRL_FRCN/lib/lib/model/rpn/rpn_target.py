from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN_Target(nn.Module):
    """ region proposal network """
    def __init__(self):
        super(_RPN_Target, self).__init__()
        
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, rpn_cls_score, rpn_bbox_pred, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)

        # generating training labels and build the rpn loss
 
        assert gt_boxes is not None

        rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

        # compute classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        rpn_label = rpn_data[0].view(batch_size, -1)

        rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
        rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
        rpn_label = Variable(rpn_label.long())
        self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
        fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

        # compute bbox regression loss
        rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
        rpn_bbox_targets = Variable(rpn_bbox_targets)

        self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])


        print(rpn_cls_score.size())

        return self.rpn_loss_cls, self.rpn_loss_box
