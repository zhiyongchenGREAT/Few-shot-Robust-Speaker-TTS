import os
import sys
import errno
import os.path as osp
import numpy as np
import torch

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def save_networks(networks, result_dir, name='', loss='', criterion=None):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)
'''

def save_networks(networks, result_dir, options, criterion=None):
    model_name = f"{options['model']}_{options['loss']}_{options['item']}_{options['cs_fish']}_{options['cs_gpt']}_{options['cs_fishgpt']}_{options['cs_my']}"
    checkpoints_dir = osp.join(result_dir, 'checkpoints')
    mkdir_if_missing(checkpoints_dir)
    
    weights = networks.state_dict()
    model_filename = osp.join(checkpoints_dir, f"{model_name}.pth")
    torch.save(weights, model_filename)

    if criterion:
        criterion_weights = criterion.state_dict()
        criterion_filename = osp.join(checkpoints_dir, f"{model_name}_criterion.pth")
        torch.save(criterion_weights, criterion_filename)

    print(f"Model saved to: {model_filename}")
    if criterion:
        print(f"Criterion saved to: {criterion_filename}")

'''
def save_GAN(netG, netD, result_dir, name=''):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = netG.state_dict()
    filename = '{}/{}_G.pth'.format(result_dir, name)
    torch.save(weights, filename)
    weights = netD.state_dict()
    filename = '{}/{}_D.pth'.format(result_dir, name)
    torch.save(weights, filename)

def load_networks(networks, result_dir, name='', loss='', criterion=None):
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    networks.load_state_dict(torch.load(filename),strict=False)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        criterion.load_state_dict(torch.load(filename))

    return networks, criterion


def save_networks_1(networks, result_dir, name='', loss='', criterion=None, embedding_model=None):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)
    if embedding_model:
        weights = embedding_model.state_dict()
        filename = '{}/checkpoints/{}_{}_eres2netv2.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)