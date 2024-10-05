import yaml
import torch
import logging
import argparse

import torch
import torch.utils.data
from torch import nn, optim

from model import VAE
from dataset import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='choose config file')
    return parser.parse_args()

def load_yaml_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

## TODO 1: 
## =================================================================
def loss_function(recon_x, x, mu, logvar,lamb, mu_att, logvar_att):
    pass

def get_batches(iterable, batch_size):
    pass

def train(epoch,lambda_kl):
    pass

def test(epoch,lambda_kl):
    pass
## =================================================================


if __name__=='main':
    args = parse_args()

    ## Load config.yaml
    config = load_yaml_config(args.config)

    ## Assign hyperparameters from config
    optimizer_choice = config['optimizer_choice']
    learning_rate = config['learning_rate']

    ## load dataset from config
    dataset = Dataset(config)

    ## load model from config
    model = VAE(config)

    ## set optimizer from config
    if (optimizer_choice=='AdamW'):
        optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
    elif(optimizer_choice=='SGD'):
        optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    else: 
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    ## TODO 2: 
    ## =================================================================
    
  
    pass