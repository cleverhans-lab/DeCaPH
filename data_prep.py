import argparse
import os
import random
import shutil

import torch
from torch.utils.data import Subset

import client
import utils
import warnings
from functools import partial
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_exp', type=str, default='data_prep')

    parser.add_argument('--seed', type=int, default=0)

    # Dataset
    parser.add_argument('--dataset', type=str, help="pancreas, xray", default="pancreas")

    parser.add_argument('--unique_patients', type=int, default=0)
    parser.add_argument('--xray_views', type=str, default='AP-PA')
    parser.add_argument('--only_include', type=str, nargs="+",
                        default=['Atelectasis', 'Effusion', 'Cardiomegaly', 'No Finding'])
    # Kfold
    parser.add_argument('--kfold', type=int, default=5,
                        help='kfold cv')
    # dataset path
    parser.add_argument('--recreate_data', type=int, default=1,
                        help='whether to recreate the 5-fold train test split')
    parser.add_argument('--dataset_path', type=str, default='', help='path to which the datasets are saved')
    parser.add_argument('--split_info_path', type=str, default='',
                        help="where to store the train test split info")

    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args()

    if not os.path.exists(args.split_info_path):
        args.split_info_path = f'split_info_path/{args.dataset}'
        args.save_dir = args.split_info_path
        if not os.path.exists(args.split_info_path):
            os.makedirs(args.split_info_path)

    utils.set_seed(args.seed)

    logger = utils.get_log(args)
    logger.info(vars(args))

    utils.prepare_dataset(args)



