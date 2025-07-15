import sys
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

from models import gan
from models.models import classifier_spk, classifier_spk_abn,classifier_spk_eresnet
from my_datasets.osr_dataloader_spk_n import SpeakerDataloader, SpeakerDataloader_tmp
from utils import Logger, save_networks, load_networks
from core import train, test_my, test_fused
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from score.my_scorer import score_me

parser = argparse.ArgumentParser("Training")

# dataset
parser.add_argument('--dataset', type=str, default='vox')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--model-path', type=str, default='./model_path')

# optimization
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=75)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=5, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.001, help="weight for entropy loss")
parser.add_argument('--beta_gan', type=float, default=1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='adapter')

# ohter parameters
parser.add_argument('--nz', type=int, default=128)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='SpeakerRPLv2')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--split-id', type=int, default=10, help="Split index for dataset")


def main_worker(options):
    torch.manual_seed(options['seed'])
    split_id = options['split_id']
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

    Data = SpeakerDataloader(
        known=list(range(140)), # 40 targets + 40 unknown spk + 40 adaptive anchors
        train_root=f'./data/vox1_eres/train{split_id}',
        test_root=f'./data/vox1_eres/test{split_id}',
        batch_size=options['batch_size']
    )
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    options['num_classes'] = Data.num_classes

    # Model
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

    # Log
    # model_path = os.path.join(options['outf'], 'models', options['dataset'])
    model_path = options['model_path']
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # file_name = '{}_{}'.format(options['model'], options['loss'])
    # file_name = 'adapter'

    # Evaluation
    if options['eval']:
        net, criterion = load_networks(net, model_path, name=options['model'], loss=options['loss'], criterion=criterion)
        results, _pred_emb, _labels, _pred_emb_u,_pred_k,_pred_u = test_my(net, criterion, testloader, outloader, epoch=0, **options)
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
            results, _pred_emb, _labels, _pred_emb_u,_pred_k,_pred_u = test_my(net, criterion, testloader, outloader, **options)
            eer = calculate_eer_metrics(_pred_k, _labels, _pred_u)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))
            print_metrics(eer)

            save_networks(net, model_path, criterion=criterion)
            print("model path: ", model_path)
      
        if options['stepsize'] > 0: scheduler.step()

        # Save CPs and RPs
        cp_dict = criterion.Dist2.centers.detach().cpu().numpy()
        rp_dict = criterion.Dist.centers.detach().cpu().numpy()
        
        if epoch == options['max_epoch'] - 1:
            np.save(f"./CP_RP/CP__split{split_id}_seed{options['seed']}_epoch74.npy", cp_dict)
            np.save(f"./CP_RP/RP__split{split_id}_seed{options['seed']}_epoch74.npy", rp_dict)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return {
    'seed': options['seed'],
    'ACC': results['ACC'],
    'AUROC': results['AUROC'],
    'OSCR': results['OSCR'],
    'EER': eer * 100,
    'Min_C': min_c,
    'TNR': results['TNR'],
    'DTACC': results['DTACC'],
    'AUIN': results['AUIN'],
    'AUOUT': results['AUOUT']
}, _pred_k, _labels, _pred_u


def run_experiment():
    all_preds_k = []
    all_labels = []
    all_preds_u = []
    all_metrics = []

    seed_list = list(range(30))
    for seed in seed_list: 
        print(f"Running experiment with seed {seed}")
        args = parser.parse_args()
        options = vars(args)
        options.update({
            'seed': seed,
            'gpu': '0',
            'batch_size': 16,
            'lr': 0.01,
            'max_epoch':75,
            'stepsize': 30,
            'eval_freq': 75,
            'use_cpu': False,
            'dataset': 'vox',
            'loss': 'SpeakerRPLv2',
            'outf': './log',
            'split_id': 1,
        })

        split_id = options['split_id']
        
        result_dict, _pred_k, _labels, _pred_u = main_worker(options)
        result_dict['pred_k'] = _pred_k
        result_dict['labels'] = _labels
        result_dict['pred_u'] = _pred_u
        all_metrics.append(result_dict)

        all_preds_k.append(_pred_k)
        all_labels.append(_labels)
        all_preds_u.append(_pred_u)


    # Step 1: CP/RP eigenvalue variance calculation
    points_dir = "./CP_RP"
    cp_vars = []
    rp_vars = []

    for seed in seed_list:
        cp_path = os.path.join(points_dir, f"CP__split{split_id}_seed{seed}_epoch74.npy")
        rp_path = os.path.join(points_dir, f"RP__split{split_id}_seed{seed}_epoch74.npy")


        cp = np.load(cp_path)[:55] # 55 = 5 targets + 50 unknown spk
        rp = np.load(rp_path)[:55]

        cp_sim = cp @ cp.T
        rp_sim = rp @ rp.T

        cp_eigvals = np.linalg.eigvalsh(cp_sim)
        rp_eigvals = np.linalg.eigvalsh(rp_sim)

        cp_vars.append(np.var(cp_eigvals))
        rp_vars.append(np.var(rp_eigvals))

    cp_ranks = np.argsort(cp_vars)
    rp_ranks = np.argsort(rp_vars)

    cp_rank_dict = {seed_list[i]: int(np.where(cp_ranks == i)[0][0]) for i in range(len(seed_list))}
    rp_rank_dict = {seed_list[i]: int(np.where(rp_ranks == i)[0][0]) for i in range(len(seed_list))}

    # Step 2: Save results to CSV
    csv_path = f'./result/split{split_id}.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['seed', 'ACC', 'AUROC', 'OSCR', 'EER', 'Min_C', 'TNR', 'DTACC', 'AUIN', 'AUOUT',
                        'CP_eigenval_Rank', 'RP_eigenval_Rank', 'filtered_seeds']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, metric in enumerate(all_metrics):
            seed = metric['seed']
            writer.writerow({
                'seed': seed,
                'ACC': round(metric['ACC'], 3),
                'AUROC': round(metric['AUROC'], 3),
                'OSCR': round(metric['OSCR'], 3),
                'EER': round(metric['EER'], 3),
                'Min_C': round(metric['Min_C'], 3),
                'TNR': round(metric['TNR'], 3),
                'DTACC': round(metric['DTACC'], 3),
                'AUIN': round(metric['AUIN'], 3),
                'AUOUT': round(metric['AUOUT'], 3),
                'CP_eigenval_Rank': cp_rank_dict[seed],
                'RP_eigenval_Rank': rp_rank_dict[seed]
            })

    # Step 3: filtered rank
    cp_ranks = np.argsort(cp_vars) 
    rp_ranks = np.argsort(rp_vars) 

    cp_ranks_filtered = cp_ranks[:-10]  
    rp_ranks_filtered = rp_ranks[:-10] 

    filtered_seeds = list(set(cp_ranks_filtered) & set(rp_ranks_filtered))
    print(f"Remaining seeds after removing the last 10 CP and RP ranks: {filtered_seeds}")


    # Step 4: fusion
    filtered_pred_k = [all_metrics[i]['pred_k'] for i in filtered_seeds]
    filtered_pred_u = [all_metrics[i]['pred_u'] for i in filtered_seeds]

    avg_pred_k = np.mean(np.stack(filtered_pred_k, axis=0), axis=0)
    avg_pred_u = np.mean(np.stack(filtered_pred_u, axis=0), axis=0)

    _labels = all_metrics[0]['labels']

    # Step 5: evaluation
    final_results = test_fused(avg_pred_k, avg_pred_u, _labels)

    print("\n==> Fused Evaluation:")
    for k, v in final_results.items():
        print(f"{k}: {v:.2f}")

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            'seed': 'fused_topN',
            'ACC': round(final_results['ACC'], 3),
            'AUROC': round(final_results['AUROC'], 3),
            'OSCR': round(final_results['OSCR'], 3),
            'EER': round(final_results['EER'], 3),
            'Min_C': round(final_results['Min_C'], 3),
            'TNR': round(final_results['TNR'], 3),
            'DTACC': round(final_results['DTACC'], 3),
            'AUIN': round(final_results['AUIN'], 3),  
            'AUOUT': round(final_results['AUOUT'], 3),
            'CP_eigenval_Rank': '-', 'RP_eigenval_Rank': '-',
            'filtered_seeds': ','.join(map(str, filtered_seeds))
        })

 
run_experiment()