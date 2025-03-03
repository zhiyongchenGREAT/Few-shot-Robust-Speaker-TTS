import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
# from utils import accuracy

class aamsoftmax(nn.Module):
    def __init__(self, **options):
        super(aamsoftmax, self).__init__()

        self.test_normalize = True
        
        self.m = options.get('margin', math.pi / 6)
        self.s = options.get('scale', 15)
        self.in_feats = options['feat_dim']
        self.weight = torch.nn.Parameter(torch.FloatTensor(options['num_classes'], self.in_feats), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = options.get('easy_margin',False)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, y=None,label=None):
        if label is None:
            return F.linear(F.normalize(x), F.normalize(self.weight)), None
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        if label.dim() > 1:
            label = label.argmax(dim=1)
        # print("label:",label)

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        label = label.to(torch.int64)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s


        loss    = self.ce(output, label)
        # prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return output, loss