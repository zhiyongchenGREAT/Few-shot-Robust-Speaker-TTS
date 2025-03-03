import torch
import torch.nn as nn
import torch.nn.functional as F

class logitnorm(nn.Module):
    def __init__(self, **options):
        
        super(logitnorm, self).__init__()
        self.feat_dim = options['feat_dim']
        self.num_classes = options['num_classes']
        self.margin = options.get('margin', 1)
        self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, y=None, labels=None):
        x = F.normalize(x, p=2, dim=1)  
        weight = F.normalize(self.weight, p=2, dim=1)
        cos_theta = torch.mm(x, weight.t())
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        logits = F.softmax(cos_theta, dim=1)
        logits = logits / torch.norm(logits, dim=1, keepdim=True) 
        if labels is None:
            return logits, 0

        labels = labels.long()
        target_cos_theta = cos_theta[torch.arange(0, labels.size(0)), labels]
        modified_cos_theta = torch.cos(self.margin * torch.acos(target_cos_theta))
        # print(self.margin)
        cos_theta[torch.arange(0, labels.size(0)), labels] = modified_cos_theta

        log_probs = F.log_softmax(cos_theta, dim=1)
        loss = F.nll_loss(log_probs, labels)

        return logits, loss
