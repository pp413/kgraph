import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class Loss(Module):

    def __init__(self, flags='triple'):
        super(Loss, self).__init__()

        assert flags in ['triple', 'pair'], 'flags must be triple or pair'
        self.flags = flags

    def forward(self, score, label):
        if self.flags == 'triple':
            score = score.flatten()
            label = label.flatten()
            batch_size = (1 - torch.abs(label)).sum().long()
            pos_score = score[:batch_size]
            neg_score = score[batch_size:]
            
            pos_score = pos_score.view(-1, batch_size).permute(1, 0)
            neg_score = neg_score.view(-1, batch_size).permute(1, 0)
            return self.forward_triple(pos_score, neg_score)
        elif self.flags == 'pair':
            return self.forward_pair(score, label)
        else:
            return 'Error: flags must be triple or pair'