import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

from models.resnet import resnet50, resnet18
from utils.calc_acc import calc_acc

from layers import TripletLoss
from layers import CSLoss
from layers import CenterLoss
from layers import cbam
from layers import NonLocalBlockND
from layers import DualBNNeck
from layers.module.part_pooling import TransformerPool, SAFL


class Baseline(nn.Module):
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning

        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)
            D = 2048
        elif backbone == "resnet18":
            self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)
            D = 512

        self.base_dim = D
        self.dim = D
        self.k_size = kwargs.get('k_size', 8)
        self.part_num = kwargs.get('num_parts', 7)
        self.dp = kwargs.get('dp', "l2")
        self.dp_w = kwargs.get('dp_w', 0.5)
        self.cs_w = kwargs.get('cs_w', 1.0)
        self.margin1 = kwargs.get('margin1', 0.01)
        self.margin2 = kwargs.get('margin2', 0.7)

        self.attn_pool = SAFL(part_num=self.part_num)
        self.bn_neck = DualBNNeck(self.base_dim + self.dim * self.part_num)

        self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data
        self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

        self.KL_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.weight_KL = kwargs.get('weight_KL', 2.0)
        self.update_rate = kwargs.get('update_rate', 0.2)
        self.update_rate_ = self.update_rate

        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num , num_classes, bias=False)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)
    
    def forward(self, inputs, labels=None, **kwargs):
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        global_feat = self.backbone(inputs)

        b, c, w, h = global_feat.shape

        part_feat, attn = self.attn_pool(global_feat)
        global_feat = global_feat.mean(dim=(2, 3))
        feats = torch.cat([part_feat, global_feat], dim=1)

        if self.training:
            masks = attn.view(b, self.part_num, w*h)
            if self.dp == "cos":
                loss_dp = torch.bmm(masks, masks.permute(0, 2, 1))
                loss_dp = torch.triu(loss_dp, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp += -masks.mean() + 1 
            elif self.dp == "l2":
                loss_dp = 0 
                for i in range(self.part_num):
                    for j in range(i+1, self.part_num):
                        loss_dp += ((((masks[:, i] - masks[:, j]) ** 2).sum(dim=1) /(18 * 9)) ** 0.5).sum()
                loss_dp = - loss_dp / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp *= self.dp_w
        if not self.training:
            feats = self.bn_neck(feats, sub)
            return feats
        else:
            return self.train_forward(feats, labels, loss_dp, sub, **kwargs)

    def train_forward(self, feat, labels, loss_dp, sub, **kwargs):
        metric = {}

        loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)
        feat = self.bn_neck(feat, sub)
    
        logits = self.classifier(feat)
        loss_id = self.ce_loss_fn(logits.float(), labels)
        tmp = self.ce_loss_fn(logits.float(), labels)
        metric.update({'ce': tmp.data})
        
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        
        logits_v = self.visible_classifier(feat[sub == 0])
        loss_id += self.ce_loss_fn(logits_v.float(), labels[sub == 0])
        logits_i = self.infrared_classifier(feat[sub == 1])
        loss_id += self.ce_loss_fn(logits_i.float(), labels[sub == 1])
        
        logits_m = torch.cat([logits_v, logits_i], 0).float()
        with torch.no_grad():
            self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                + self.infrared_classifier.weight.data * self.update_rate
            self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                + self.visible_classifier.weight.data * self.update_rate

            logits_v_ = self.infrared_classifier_(feat[sub == 0])
            logits_i_ = self.visible_classifier_(feat[sub == 1])
            logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()

        loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1)) 

        metric.update({'id': loss_id.data})
        metric.update({'cs': loss_cs.data})
        metric.update({'dp': loss_dp.data})

        loss = loss_id + loss_cs * self.cs_w + loss_dp * self.dp_w 

        return loss, metric