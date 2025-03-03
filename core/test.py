import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation



def test(net, criterion, testloader, outloader, epoch=None, save_scores_path=None, save_results_path=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                
                # print("logits shape: ",logits.shape)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())

    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    # print("x1: ",x1)
    # print("x2: ",x2)
    results = evaluation.metric_ood(x1, x2)['Bas']

    _oscr_score = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_score * 100.


    return results

'''
def test(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            # print(labels)
            with torch.set_grad_enabled(False):
                # print(data.shape)
                x, y = net(data, True)
                # print(x.shape)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            # print(labels)
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results

'''

def test_eer(net, criterion, testloader, outloader, epoch=None, save_scores_path=None, save_results_path=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    score_list = []
    label_list = []

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

                scores = F.softmax(logits, dim=1)  
                score_list.append(scores.cpu().numpy())
                
                one_hot_labels = F.one_hot(labels, num_classes=logits.size(1))
                label_list.append(one_hot_labels.cpu().numpy())

        for data, labels in outloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logits, _ = criterion(x, y)

                scores = F.softmax(logits, dim=1)  
                score_list.append(scores.cpu().numpy())
                
                # one_hot_labels = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
                num_classes = logits.size(1)
                one_hot_labels = torch.zeros(logits.size(0), num_classes, device=logits.device)

                # Check bounds explicitly
                if labels.max() >= num_classes or labels.min() < 0:
                    raise ValueError(f"Labels out of bounds. Max: {labels.max()}, Num Classes: {num_classes}")

                one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)
                                
                label_list.append(one_hot_labels.cpu().numpy())

    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    score_list = np.concatenate(score_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)

    return score_list, label_list


# def test_my(net, criterion, testloader, outloader, epoch=None, **options):
#     net.eval()
#     correct, total = 0, 0

#     torch.cuda.empty_cache()

#     _pred_k, _pred_u, _labels = [], [], []
#     _pred_emb, _pred_emb_u = [], []
#     _outlier_labels = []  

#     with torch.no_grad():
#         for data, labels in testloader:
#             if options['use_gpu']:
#                 data, labels = data.cuda(), labels.cuda()
#             with torch.set_grad_enabled(False):
#                 x, y = net(data, True)
#                 logits, _ = criterion(x, y)
#                 predictions = logits.data.max(1)[1]
#                 total += labels.size(0)
#                 correct += (predictions == labels.data).sum()

#                 _pred_emb.append(x.data.cpu().numpy())
#                 _pred_k.append(logits.data.cpu().numpy())
#                 _labels.append(labels.data.cpu().numpy())

#         for batch_idx, (data, labels) in enumerate(outloader):
#             if options['use_gpu']:
#                 data, labels = data.cuda(), labels.cuda()
#             with torch.set_grad_enabled(False):
#                 x, y = net(data, True)
#                 logits, _ = criterion(x, y)
#                 _pred_u.append(logits.data.cpu().numpy())
#                 _pred_emb_u.append(x.data.cpu().numpy())
#                 _outlier_labels.append(labels.data.cpu().numpy()) 

#     acc = float(correct) * 100. / float(total)
#     print('Acc: {:.5f}'.format(acc))

#     _pred_k = np.concatenate(_pred_k, 0)
#     _pred_u = np.concatenate(_pred_u, 0)
#     _labels = np.concatenate(_labels, 0)
#     _outlier_labels = np.concatenate(_outlier_labels, 0) 

#     x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
#     results = evaluation.metric_ood(x1, x2)['Bas']

#     _oscr_score = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

#     results['ACC'] = acc
#     results['OSCR'] = _oscr_score * 100.

#     return results, _pred_emb, _labels, _pred_emb_u, _outlier_labels


def test_my(net, criterion, testloader, outloader, epoch=None, **options):
    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []
    _pred_emb,_pred_emb_u = [], []
    _out_labels = []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            # print(labels)
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
            # print(labels)
            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                # x, y = net(data, return_feature=True)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())
                _pred_emb_u.append(x.data.cpu().numpy())
                _out_labels.append(labels.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    _out_labels = np.concatenate(_out_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results, _pred_emb, _labels, _pred_emb_u,_pred_k,_pred_u,_out_labels