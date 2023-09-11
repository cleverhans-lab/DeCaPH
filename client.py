import copy
import os
import random
import shutil
import math

import numpy as np
import torch
from torch import optim as optim
from torch.utils.data import Subset, DataLoader

import custom_model
import utils
from utils import get_dataset, get_loss_func, evaluate_model, print_and_log, save_state, load_state, \
    get_or_load_sequence, print_save_epoch_results
import warnings
from itertools import chain
from tqdm import tqdm
import time


def get_private_trainloader_and_public_testloader(args, id):
    if "xray" in args.dataset: # xray_dataset
        from datasets_utils.xray.xray_utils import all_xray_ids
        private_dataset = get_dataset(args, train=True, dataset_id=all_xray_ids[id],
                                      get_list=False)
        testdata = get_dataset(args, train=False, dataset_id='all',
                               get_list=False)
        private_trainloader = torch.utils.data.DataLoader(private_dataset,
                                                   batch_size=args.batch_size[id],
                                                          num_workers=args.num_workers,
                                                          pin_memory=True, shuffle=True
                                                   )
        testloader = torch.utils.data.DataLoader(testdata,
                                                   batch_size=args.eval_batch_size,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=False
                                                   )
        return private_trainloader, testloader, private_dataset, testdata

    elif args.dataset.lower() == "pancreas":
        from datasets_utils.pancreas.pancreas_utils import all_study_ids
        private_dataset = get_dataset(args, train=True, dataset_id=all_study_ids[id],
                                      get_list=False)
        testdata = get_dataset(args, train=False, dataset_id='all',
                               get_list=False)
        private_trainloader = DataLoader(private_dataset,
                                         batch_size=args.batch_size[id],
                                         num_workers=args.num_workers,
                                         pin_memory=True, shuffle=True)
        testloader = torch.utils.data.DataLoader(testdata, batch_size=args.eval_batch_size,
                                                 num_workers=args.num_workers, pin_memory=True)
        return private_trainloader, testloader, private_dataset, testdata
    else:
        raise NotImplementedError(f"{args.dataset} not implemented")


def train_client_models(args, list_clients_id, from_global_epoch=0, all_clients={},
                        sequence_dict={}):
    """
    trains all selected clients and stores their (local) updates
    @param args:
    @param list_clients_id: the selected clients' id's for this run
    @return: None
    """
    print_and_log(args, f"global {from_global_epoch} clients {list_clients_id} ", 1)
    for id in list_clients_id:
        all_clients[id].net.train()
        if all_clients[id].freeze_running_stats:
            custom_model.freeze_bn(all_clients[id].net)
        all_clients[id].get_private_trainloader(sequence=sequence_dict[id])
        all_clients[id].train(from_global_epoch)


def train_epoch(model, private_trainloader, criterion,
                optimizer, privacy_engine,
                dataset, device, freeze_running_stats=False,
                args=None, scheduler=None):
    avg_loss = []

    optimizer.zero_grad()
    private_trainloader_t = tqdm(private_trainloader)

    for i, data in enumerate(private_trainloader_t):
        if privacy_engine is not None:
            # check if next step would exceed target budget
            optimizer.privacy_engine.steps += 1
            next_epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()
            optimizer.privacy_engine.steps -= 1
            if next_epsilon >= args.target_budget:
                print_and_log(args, f"next step eps {next_epsilon} > target eps {args.target_budget}", 1)
                break

        model.train()
        if freeze_running_stats:
            custom_model.freeze_bn(model)
        inputs, labels = utils.get_batch_data(data, dataset, device)

        if privacy_engine is None:
            outputs = utils.get_output(inputs, model, dataset, device)
            loss = utils.get_loss(outputs, labels, criterion, dataset, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            for num_pseudo_batch in range(math.ceil(len(inputs)/args.physical_batch_size)):
                torch.cuda.empty_cache()
                start_ind = num_pseudo_batch * args.physical_batch_size
                end_ind = (num_pseudo_batch + 1) * args.physical_batch_size
                input_temp = inputs[start_ind: end_ind].to(device)
                label_temp = labels[start_ind: end_ind].to(device)
                outputs = utils.get_output(input_temp, model, dataset, device)
                loss = utils.get_loss(outputs, label_temp, criterion, dataset, device)
                loss.backward()
                if (num_pseudo_batch + 1) == math.ceil(len(inputs)/args.physical_batch_size):
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.virtual_step()
                torch.cuda.empty_cache()

        avg_loss.append(loss.detach().cpu().numpy())
        private_trainloader_t.set_postfix(loss=f"{np.mean(avg_loss):4.4f}")
        torch.cuda.empty_cache()

    if scheduler is not None:
        scheduler.step()


def sample_clients_id(args):
    """
    sample the clients to train for this run
    @param args:
    @return: a list of selected clients' id's
    """
    all_clients_id = list(range(args.num_clients))
    m = max(int(len(all_clients_id) * args.sample_clients_ratio), 1)
    list_client_id = random.sample(all_clients_id, int(m))
    args.m = m
    args.sum_selected_clients_weights = np.sum([args.client_data_split_ratio[i] for i in list_client_id])
    args.logger.info(f"num clients sampled: {args.m}")
    args.logger.info(f"client ids selected for this aggregation round: {list_client_id}")
    return list_client_id

