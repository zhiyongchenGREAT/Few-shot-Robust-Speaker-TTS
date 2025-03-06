import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import numpy as np
from models.models import classifier_spk_eresnet
from my_datasets.osr_dataloader_spk_n import SpeakerDataloader
from utils import Logger, save_networks, load_networks
from core import train, test, test_my
from score.my_scorer import score_me
from score.evaluate_metrics import calculate_eer_metrics, print_metrics

parser = argparse.ArgumentParser("Training")

# dataset
parser.add_argument('--dataset', type=str, default='ESD')
parser.add_argument('--train-root', type=str, default='./data/ESD/train1_full_controlling(14)')
parser.add_argument('--test-root', type=str, default='./data/ESD/test1')
parser.add_argument('--outf', type=str, default='./log')

# optimization
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=5, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.001, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')
parser.add_argument('--loss', type=str, default='SpeakerRPL')
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--ns', type=int, default=1)

# other parameters
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)

def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    train_root = options['train_root']
    if os.path.exists(train_root):
        speaker_folders = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
        num_speakers = len(speaker_folders)
    else:
        print(f"train_root warning")
    
    Data = SpeakerDataloader(known=list(range(num_speakers)), train_root=options['train_root'], 
                            test_root=options['test_root'], batch_size=options['batch_size'])
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format('classifier_spk_eresnet'))
    net = classifier_spk_eresnet(num_classes=options['num_classes'])

    feat_dim = 192

    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    # Log
    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_name = '{}_{}'.format(options['model'], options['loss'])

    # Evaluation
    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results, _pred_emb, _labels, _pred_emb_u,_pred_k,_pred_u,_out_labels = test_my(net, criterion, testloader, outloader, epoch=0, **options)
        eer = calculate_eer_metrics(_pred_k, _labels, _pred_u)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
        print_metrics(eer)
        return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,80,120,150])

    # Training
    start_time = time.time()
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results, _pred_emb, _labels, _pred_emb_u,_pred_k,_pred_u,_out_labels = test_my(net, criterion, testloader, outloader, **options)
            eer = calculate_eer_metrics(_pred_k, _labels, _pred_u)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            print_metrics(eer)

            save_networks(net, model_path, file_name, criterion=criterion)
            print("model path: ", model_path)

        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    res = main_worker(options)