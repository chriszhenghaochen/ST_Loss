import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from loss import MMD, JMMD 

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        #transfer weight
        self.transfer_weight = Variable(torch.Tensor([cfg.Trade_Off]).cuda(), requires_grad=True)


    def FRCN(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        #store pool 5 information
        pool5_flat_s = base_feat.view(base_feat.size(0), -1)

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # print('base_feat ', base_feat.size())
        # print('fc7 ',pooled_feat.size())
        # print('labels ', rois_label.size())

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, pooled_feat


    def forward(self, im_data, im_info, gt_boxes, num_boxes, t_im_data = None, t_im_info = None, t_gt_boxes = None, t_num_boxes = None, transfer = False):

        rois, cls_prob, \
        bbox_pred, rpn_loss_cls, \
        rpn_loss_bbox, RCNN_loss_cls, \
        RCNN_loss_bbox, rois_label, pooled_feat = self.FRCN(im_data, im_info, gt_boxes, num_boxes)

        t_rois, t_cls_prob, \
        t_bbox_pred, t_rpn_loss_cls, \
        t_rpn_loss_bbox, t_RCNN_loss_cls, \
        t_RCNN_loss_bbox, t_rois_label, t_pooled_feat = 0, 0, 0, 0, 0, 0, 0, 0, 0

        transfer_loss = 0

        if self.training and transfer:
            t_rois, t_cls_prob, \
            t_bbox_pred, t_rpn_loss_cls, \
            t_rpn_loss_bbox, t_RCNN_loss_cls, \
            t_RCNN_loss_bbox, t_rois_label, t_pooled_feat = self.FRCN(t_im_data, t_im_info, t_gt_boxes, t_num_boxes)

            ids_s = torch.LongTensor(1).cuda()
            ids_t = torch.LongTensor(1).cuda()

            # random select
            if cfg.TRANSFER_SELECT == 'RANDOM':
                perm = torch.randperm(pooled_feat.size(0))
                ids_s = perm[:pooled_feat.size(0)/8].cuda()
                ids_t = ids_s

            # select positive sample and predicted postive sample
            elif cfg.TRANSFER_SELECT == 'CONDITION':
                ids_s = torch.range(0, pooled_feat.size(0)/8 - 1)
                ids_s = torch.Tensor.long(ids_s).cuda()
                ids_t = ids_s
                # _, ids_t = torch.topk(t_cls_score[:,0], pooled_feat.size(0)/8)
                # ids_t = ids_t.cuda()

            elif cfg.TRANSFER_SELECT == 'POSITIVE':
                ids_s = torch.nonzero(rois_label.data)
                ids_t = torch.nonzero(t_rois_label.data)

                ids_size = torch.min(torch.IntTensor([ids_s.size()[0], ids_t.size()[0]]))
                ids_s = torch.squeeze(ids_s[:ids_size]).cuda()
                ids_t = torch.squeeze(ids_t[:ids_size]).cuda()

            # print('source ', ids_s)
            # print('target ', ids_t)


            # calculate MMD pr JMMD loss
            if cfg.TRANSFER_LOSS == 'MMD':
                transfer_loss = MMD(pooled_feat[ids_s], t_pooled_feat[ids_t])

            elif cfg.TRANSFER_LOSS == 'JMMD':
                transfer_loss = JMMD([pooled_feat[ids_s], cls_prob[ids_s], bbox_pred[ids_s]], [t_pooled_feat[ids_t], t_cls_prob[ids_t], t_bbox_pred[ids_t]])

            elif cfg.TRANSFER_LOSS == 'feature_MMD':
                transfer_loss = MMD(pool5_flat_s, pool5_flat_t)
        
            # #debug session
            # print('source ', pooled_feat[ids_s].size())
            # print('target ', t_pooled_feat[ids_t].size())
            # #debug done

            transfer_loss = transfer_loss*(self.transfer_weight.expand_as(transfer_loss))
        #-----------------------------------Tranfer learninig Done------------------------------#
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, transfer_loss, \
               t_rois, t_cls_prob, t_bbox_pred, t_rpn_loss_cls, t_rpn_loss_bbox, t_RCNN_loss_cls, t_RCNN_loss_bbox, t_rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
