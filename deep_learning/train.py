import torch
import argparse
from stupidnewstuff.decomper import Decomper
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from utils import dataloader
from pathlib import Path


def main(args):
    pl.seed_everything(args.seed)

    # Load data
    df = pd.read_csv(Path(args.data_folder + args.data_file), index_col=[0])

    # Experiment parameters
    continuous_cols_idx = [int(x) for x in str.split(args.continuous_cols_idx, ',')]
    categorical_cols_idx = [int(x) for x in str.split(args.categorical_cols_idx, ',')] \
        if args.categorical_cols_idx != '' else []
    target_cols_idx =  [int(x) for x in str.split(args.target_cols_idx, ',')]
    input_dim = len(continuous_cols_idx + categorical_cols_idx)
    
    # Univariate or multivariate dataset and laoders
    dataset = dataloader.TSDataset(df, continuous_cols_idx, categorical_cols_idx, target_cols_idx,
                                    args.input_len, args.forecast_horizon)
    train_loader, test_loader = dataset.get_loaders(batch_size=args.batch_size)

    # Define models and training objects
    model = Decomper(input_dim, args.hidden_dim, args.output_dim, args.num_layers)

    trainer = pl.Trainer(gpus=args.gpu_num, max_epochs=args.epochs, check_val_every_n_epoch=args.check_val_every_n_epoch)

    # Fit model
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=test_loader)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Forecasting NNs')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='data/') 
    parser.add_argument('--data_file', type=str, default='air_quality_simplified.csv')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--target_cols_idx', type=str, default='3')
    parser.add_argument('--continuous_cols_idx', type=str, default='0,1,2')
    parser.add_argument('--categorical_cols_idx', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--input_len', type=int, default=20)
    parser.add_argument('--forecast_horizon', type=int, default=4)
    parser.add_argument('--output_dim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    
    args = parser.parse_args()
    main(args)