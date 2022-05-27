import torch
from torch.nn import Module

class Loss(Module):

    def __init__(self, element_type='triple'):
        super(Loss, self).__init__()

        assert element_type in ['triple', 'pair'], 'element_type must be triple or pair'
        self.element_type = element_type

    def forward(self, score, label):
        if self.element_type == 'triple':
            score = score.flatten()
            label = label.flatten()
            batch_size = (1 - torch.abs(label)).sum().long()
            pos_score = score[:batch_size]
            neg_score = score[batch_size:]
            
            pos_score = pos_score.view(-1, batch_size).permute(1, 0)
            neg_score = neg_score.view(-1, batch_size).permute(1, 0)
            return self.forward_triple(pos_score, neg_score)
        elif self.element_type == 'pair':
            return self.forward_pair(score, label)
        else:
            return 'Error: element_type must be triple or pair'