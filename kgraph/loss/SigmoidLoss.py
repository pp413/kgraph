import torch
import torch.nn as nn
import torch.nn.functional as F

from .Loss import Loss

class SigmoidLoss(Loss):

    def __init__(self, adv_temperature=None, element_type='triple'):
        super(SigmoidLoss, self).__init__(element_type=element_type)
        self.criterion = nn.LogSigmoid()

        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def get_weights(self, neg_score):
        return F.softmax(neg_score*self.adv_temperature, dim=-1).detach()
    
    def forward_triple(self, pos_score, neg_score):
        if self.adv_flag:
            return -(self.criterion(pos_score).mean() + (self.get_weights(neg_score) * self.criterion(-neg_score)).sum(dim=-1).mean()) / 2
        else:
            return -(self.criterion(pos_score).mean() + self.criterion(-neg_score).mean()) / 2
    
    def forward_pair(self, score, label):
        origin_label = label.clone()
        label = label.round()
        if self.adv_flag:
            neg_softmax = self.get_weights(score) * (1 - label)
            neg_softmax = neg_softmax / neg_softmax.sum(dim=-1, keepdims=True)
            pos_loss = (self.criterion(score) * label * origin_label).mean()
            neg_loss = (neg_softmax * (self.criterion(-score) * (1-label) * (1 - origin_label))).sum(dim=-1).mean()
            return -(pos_loss + neg_loss) / 2
        else:
            pos_loss = (self.criterion(score) * label * origin_label).mean()
            neg_loss = (self.criterion(-score) * (1-label) * (1 - origin_label)).mean()
            return -(pos_loss + neg_loss) / 2