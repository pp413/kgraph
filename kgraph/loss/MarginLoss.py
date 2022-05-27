import torch
import torch.nn as nn
import torch.nn.functional as F

from .Loss import Loss

class MarginLoss(Loss):

    def __init__(self, adv_temperature=None, margin=6.0, element_type='triple'):
        super(MarginLoss, self).__init__(element_type=element_type)
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        if adv_temperature != None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False
    
    def get_weights(self, neg_score):
        return F.softmax(-neg_score*self.adv_temperature, dim=-1).detach()
    
    def forward_triple(self, pos_score, neg_score):
        if self.adv_flag:
            return (self.get_weights(neg_score) * torch.max(pos_score - neg_score, -self.margin)).sum(dim=-1).mean() + self.margin
        else:
            return (torch.max(pos_score - neg_score, -self.margin)).mean() + self.margin
    
    def forward_pair(self, score, label):
        label = label.round()
        if self.adv_flag:
            neg_softmax = self.get_weights(score) * (1 - label)
            neg_softmax = neg_softmax / neg_softmax.sum(dim=-1, keepdims=True)
            loss = torch.max((score * label).mean(dim=-1, keepdims=True) - score * (1 - label), -self.margin)
            return (neg_softmax * loss).sum(dim=-1).mean() + self.margin
        else:
            return (torch.max((score * label).mean(dim=-1, keepdims=True) - score * (1 - label), -self.margin)).mean() + self.margin