import torch
import torch.nn as nn
import numpy as np



class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss().cuda()

    def get_metric_loss(self, x, ref):
        '''
        :param x: (bs, n_rkhs)
        :param ref: (bs, n_rkhs, n_loc)
        :return: loss
        '''

        bs, n_rkhs, n_loc = ref.size()
        ref = ref.transpose(0, 1).reshape(n_rkhs, -1)
        score = torch.matmul(x, ref) * 64.  # (bs * n_loc, bs)
        score = score.view(bs, bs, n_loc).transpose(1, 2).reshape(bs * n_loc, bs)
        gt_label = torch.arange(bs, dtype=torch.long, device=x.device).view(bs, 1).expand(bs, n_loc).reshape(-1)
        return self.ce(score, gt_label)

    def forward(self, x, refs):
        loss = 0.
        for ref in refs:
            loss += self.get_metric_loss(x, ref)
        return loss


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        '''
        :param x: (bs, np, 3)
        :param y: (bs, np, 3)
        :return: loss
        '''

        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-6 + torch.sum(torch.pow(x - y, 2), 3)) # bs, ny, nx
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)

        return min1.mean() + min2.mean()
    

def NormalLoss(x, y):
    loss = torch.sum(torch.abs(x * y), dim=1).mean()
    return 1 - loss

