import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist


class SpeakerRPL(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(SpeakerRPL, self).__init__()
        self.use_gpu = options['use_gpu']
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'], init = "zero")
        self.points = self.Dist.centers
        self.radius = nn.Parameter(torch.Tensor(1))
        self.radius.data.fill_(0)
        self.margin_loss = nn.MarginRankingLoss(margin=1.0)

        self.Dist2 = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'], init = "zero") 

    def forward(self, x, y, labels=None, return_dist=False):
  
        dist_dot_p = self.Dist(x, center=self.points, metric='dot') 
        logits = -dist_dot_p 

        if return_dist:
            max_distances = dist_dot_p.max(dim=1).values
            return max_distances

        if labels is None:
            return logits, 0

        loss = F.cross_entropy(logits / self.temp, labels)

        # Margin loss
        center_batch = self.points[labels, :]
        _dis_known = (x - center_batch).pow(2).mean(1)
        target = torch.ones(_dis_known.size()).cuda()
        # target = torch.ones(_dis_known.size())
        loss_r = self.margin_loss(self.radius, _dis_known, target)

        dist2 = self.Dist2(x, metric='dot')
        loss2 = F.cross_entropy(-dist2 / self.temp, labels)
        center_batch2 = self.Dist.centers[labels, :]
        loss_r2 = F.mse_loss(x, center_batch2) / 2

        loss = loss + self.weight_pl * loss_r + loss2 + self.weight_pl * loss_r2
        return logits, loss

    def fake_loss(self, x):
        logits = self.Dist(x, center=self.points)
        prob = F.softmax(logits, dim=1)
        loss = (prob * torch.log(prob)).sum(1).mean().exp()

        return loss
