import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation

def test_my(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []
    _pred_emb,_pred_emb_u = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_emb.append(x.data.cpu().numpy())
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())
                _pred_emb_u.append(x.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0) if _pred_u else np.zeros_like(_pred_k) # to handle close-set case
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detection evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results, _pred_emb, _labels, _pred_emb_u,_pred_k,_pred_u


def test_fused(pred_k, pred_u, labels_k, labels_u=None):
    import sys
    sys.path.append('./FEW-SHOT-ROBUST-SPEAKER-TTS')
    from core import evaluation
    
    results = {}
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # 1. ACC
    pred_labels = np.argmax(pred_k, axis=1)
    acc = np.mean(pred_labels == labels_k) * 100
    results['ACC'] = acc

    # 2. OSCR
    oscr = evaluation.compute_oscr(pred_k, pred_u, labels_k)
    results['OSCR'] = oscr * 100

    # 3. EER / Min_C
    score_list, label_list = [], []

    num_classes = pred_k.shape[1]
    for i in range(pred_k.shape[0]):
        for cls in range(num_classes):
            score_list.append(pred_k[i][cls])
            label_list.append(1 if cls == labels_k[i] else 0)

    for i in range(pred_u.shape[0]):
        for cls in range(num_classes):
            score_list.append(pred_u[i][cls])
            label_list.append(0)

    score_list = np.array(score_list)
    label_list = np.array(label_list)

    from score.my_scorer import score_me
    config = {'p_target': [0.1], 'c_miss': 1, 'c_fa': 1}
    eer, min_c, act_c = score_me(score_list, label_list, config)

    results['EER'] = eer * 100
    results['Min_C'] = min_c

    return results