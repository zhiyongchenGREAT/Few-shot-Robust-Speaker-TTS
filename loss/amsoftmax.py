import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
# from utils import accuracy



class amsoftmax(nn.Module):
    def __init__(self, **options):
        super(amsoftmax, self).__init__()

        self.test_normalize = True
        
        self.m = options.get('margin', 0.2)
        self.s = options.get('scale', 15)
        self.in_feats = options['feat_dim']
        self.W = torch.nn.Parameter(torch.FloatTensor(self.in_feats,options['num_classes']), requires_grad=True)
        # self.W = torch.nn.Parameter(torch.randn(self.in_feats, options['num_classes']), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

    def forward(self, x, y=None, label=None):
        if label is None:
            return F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.W, p=2, dim=0).T), None
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        if label.dim() > 1:
            label = label.argmax(dim=1)

        # print("label:",label)

        # x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        # x_norm = torch.div(x, x_norm)
        x_norm = F.normalize(x, p=2, dim=1)

        # w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        # w_norm = torch.div(self.W, w_norm)
        w_norm = F.normalize(self.W, p=2, dim=0) 

        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1).to(torch.int64)
        # print(f"label_view: {label_view}")
        # print(f"costh.size(): {costh.size()}")
        if label_view.max() >= costh.size(1):
            raise ValueError(f"Label value {label_view.max()} exceeds number of classes {costh.size(1)}.")

        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss    = self.ce(costh_m_s, label)
        # prec1   = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        
        return costh_m_s,loss