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

from models.models import classifier_spk_eresnet
from my_datasets.osr_dataloader_spk_n import SpeakerDataloader
from utils import Logger, save_networks, load_networks
from core import train, test, test_my
import numpy as np

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='avsp', help="avsp")
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

# misc
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=14)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)

from score.my_scorer import score_me

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
    
    Data = SpeakerDataloader(known=list(range(17)), train_root='./data/IEMOCAP/train3+gpt3_control_00', 
                            test_root='./data/IEMOCAP/test3', batch_size=options['batch_size'])
    
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    options['num_classes'] = Data.num_classes

    print("Creating model: {}".format('classifier_spk'))
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


    model_path = os.path.join(options['outf'], 'models_iemocap', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    
    file_name = '{}_{}'.format(options['model'], options['loss'])

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    
    
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,80,120,150])

    start_time = time.time()
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        
        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

            save_networks(net, model_path, file_name, criterion=criterion)
            print(model_path)
            
            results, _pred_emb, _labels, _pred_emb_u,_pred_k,_pred_u,_out_labels = test_my(net, criterion, testloader, outloader, **options)
        
            num_samples = _pred_k.shape[0]  
            num_classes = _pred_k.shape[1]


            score_list = []
            label_list = []

            for i in range(num_samples):
                logits = _pred_k[i]  
                true_label = _labels[i]  

                for cls in range(num_classes):
                    score_list.append(logits[cls]) 
                    label_list.append(1 if cls == true_label else 0) 
                    
            num_out_samples = _pred_u.shape[0]  
            num_out_classes = _pred_u.shape[1]

            for i in range(num_out_samples):
                logits = _pred_u[i]  

                for cls in range(num_out_classes):
                    score_list.append(logits[cls])  
                    label_list.append(0) 

            score_list = np.array(score_list)
            label_list = np.array(label_list)


            configuration = {
                'p_target': [0.1],  
                'c_miss': 1,          
                'c_fa': 1                
            }
            eer, min_c, act_c = score_me(score_list, label_list, configuration)

            print(f"EER: {eer * 100:.2f}%")
            print(f"Min_C: {min_c:.3f}")
            print(f"Act_C: {act_c:.3f}")

        
        if options['stepsize'] > 0: scheduler.step()

        
        

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    results = dict()

    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join(options['outf'], 'results_iemocap', dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name = options['dataset'] + '.csv'

    res = main_worker(options)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(dir_path, file_name))