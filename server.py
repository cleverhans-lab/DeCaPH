import argparse
import copy
import os
import random
import shutil

import torch
from torch.utils.data import Subset

import client_FL
import utils
from client import train_client_models, sample_clients_id
from utils import get_dataset, evaluate_model, print_and_log
import numpy as np
import warnings
from functools import partial
from collections import defaultdict
import custom_model


warnings.filterwarnings('ignore')


def aggregate_models(args, list_clients_id, from_global_epoch=0, list_model_states=None):
    """
    iteratively adds the scaled clients' updates to the aggregate model.
    stores the current aggregate state
    @param args:
    @param list_clients_id: list of clients' id's that are selected to submit their updates
    this run
    @param device: cpu or cuda
    @return: the aggregate model that contains the updated aggregate state
    """
    with torch.no_grad():
        # args.client_data_split_ratio = args.batch_size
        aggregate_weight = copy.deepcopy(args.architecture().state_dict())
        list_models = {}
        for id in list_clients_id:
            list_models[id] = copy.deepcopy(list_model_states[id].net.state_dict())
        print(f"aggregating model weights ratios: {args.client_data_split_ratio}")
        for key in aggregate_weight.keys():
            weights = []
            weights_ratio = []

            for id in list_clients_id:
                weights.append(list_models[id][key] * args.client_data_split_ratio[id])
                weights_ratio.append(args.client_data_split_ratio[id])
            aggregate_weight[key] = sum(weights) / sum(weights_ratio)
        aggregate_model = args.architecture().to(args.device)
        aggregate_model.load_state_dict(aggregate_weight)
    return aggregate_model, aggregate_weight


def train_fl_model(args):
    """
    iteratively trains and stores the aggregate model.
    @param args:
    @return: None
    """
    testdata = get_dataset(args, train=False, dataset_id='all', get_list=False)

    testloader = torch.utils.data.DataLoader(testdata, batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                             pin_memory=True, shuffle=False)

    aggregate_model = args.architecture()
    utils.load_pretrained_state(aggregate_model, args)
    aggregate_state = copy.deepcopy(aggregate_model.state_dict())

    sequence_dict_all = defaultdict(list)
    sync_freq = {}
    for i in range(args.num_global_server_epoch):
        for j in range(args.num_clients):
            sequence = utils.get_or_load_sequence(args.batch_size[j], int(args.list_dataset_length[j]),
                                             args.num_local_client_epoch,
                                             drop_last=True if args.dp_option == "dpsgd" else False)
            num_batches_to_aggregate = int(np.ceil(len(sequence) / args.num_aggregates_per_global_epoch))
            if args.set_batch_dynamically:
                num_batches_to_aggregate = int(np.round(len(sequence) / args.num_aggregates_per_global_epoch))
            sync_freq[j] = num_batches_to_aggregate
            if len(sequence) < args.num_aggregates_per_global_epoch:
                raise ValueError(f"{args.num_aggregates_per_global_epoch} provided is larger than "
                                 f"num of batches per epoch"
                                 f"for client {j}")
            for k in range(args.num_aggregates_per_global_epoch):
                sequence_dict_all[j].append(sequence[k * num_batches_to_aggregate:
                                                     (k + 1) * num_batches_to_aggregate])

    all_clients = {}
    for client_id in range(args.num_clients):
        sequence_dict_all[client_id] = iter(sequence_dict_all[client_id])
        all_clients[client_id] = client_FL.Client(client_id, args)

    # if we want to the server to aggregate multiple times per epoch,
    # set args.num_aggregates_per_global_epoch > 1. if args.num_aggregates_per_global_epoch = 1,
    # args.num_global_aggregate_steps = args.num_global_server_epoch
    args.num_global_aggregate_steps = args.num_global_server_epoch * args.num_aggregates_per_global_epoch
    print(f"total number of aggregates: {args.num_global_aggregate_steps}")
    for i in range(args.num_global_aggregate_steps):
        torch.cuda.empty_cache()
        list_clients_id = sample_clients_id(args)
        list_clients_id_copy = copy.deepcopy(list_clients_id)
        sequence_dict = {}

        print_and_log(args, f"list of clients sampled for this round: {list_clients_id}", 1)
        for id in list_clients_id_copy:
            try:
                sequence_dict[id] = next(sequence_dict_all[id])
            except StopIteration:
                print_and_log(args, f"client {id} has used up but selected: {list_clients_id}. will be removed", 1)
                list_clients_id_copy.remove(id)

        if args.dp_option == 'dpsgd':
            for id in list_clients_id:
                try:
                    all_clients[id].optimizer.privacy_engine.steps += 1
                    next_epsilon, best_alpha = all_clients[id].optimizer.privacy_engine.get_privacy_spent()
                    all_clients[id].optimizer.privacy_engine.steps -= 1
                    if next_epsilon >= args.target_budget:
                        print_and_log(args, f"client {id} will use up its budget {next_epsilon} if train for another step"
                                            f"(target {args.target_budget}) "
                                            f"but selected: {list_clients_id}",
                                      1)
                        list_clients_id_copy.remove(id)
                except FileNotFoundError:
                    print_and_log(args, f"client {id} has no saved state", 1)
        if len(list_clients_id_copy) == 0:
            args.num_global_aggregate_steps = i
            print_and_log(args, f"no more client's data left for training {args.num_global_aggregate_steps}", 1)
            break
        print_and_log(args, f"start training global step: {i+1}", 1)

        train_client_models(args, list_clients_id_copy, from_global_epoch=i,
                            all_clients=all_clients, sequence_dict=sequence_dict)
        aggregate_model, aggregate_state = aggregate_models(args, list_clients_id_copy, from_global_epoch=i,
                                           list_model_states=all_clients)

        for client_id in range(args.num_clients):
            all_clients[client_id].load_aggregate_state(aggregate_state)
        if (i+1)%args.save_freq_epoch == 0:
            torch.save({'net': aggregate_state},
                       f"{args.save_dir}/aggregate_{args.num_global_aggregate_steps}.pt")

    torch.save({'net': aggregate_state},
               f"{args.save_dir}/aggregate_{args.num_global_aggregate_steps}.pt")

    eps_list, best_alpha_list, delta_list = [], [], []
    for j in range(len(all_clients)):
        eps, best_alpha, delta = None, None, None
        if args.dp_option == 'dpsgd':
            state = all_clients[j].save()
            eps, best_alpha = state['eps'], state['best_alpha']
            try:
                delta = state['delta']
            except:
                delta = None
        eps_list.append(eps)
        best_alpha_list.append(best_alpha)
        delta_list.append(delta)
    all_y_pred, all_y_val = evaluate_model(args.dataset, aggregate_model, testloader)
    utils.print_save_epoch_results(args, args.num_global_aggregate_steps-1, None, None, None, None, all_y_val, all_y_pred,
                                   name="all", epsilon=eps_list, best_alpha={"best_alpha": best_alpha_list,
                                                                             "delta": delta_list})

    if args.use_keep:
        mia_testloader = utils.get_mia_testloader(args)
        utils.save_mia_scores(aggregate_model, args, args.num_global_aggregate_steps, mia_testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_model_state', type=str, default="",
                        help='path to the initial state for transfer learning')
    parser.add_argument('--num_workers', type=int, default=0)
    # FL parameters
    parser.add_argument('--num_global_server_epoch', type=int, default=2,
                        help='number of epochs the aggregate model get updated')
    parser.add_argument('--num_aggregates_per_global_epoch', type=int, default=1,
                        help='number of aggregation performed for each global epoch')
    parser.add_argument('--num_local_client_epoch', type=int, default=1,
                        help='number of epochs the client model gets updated each ')
    parser.add_argument('--sample_clients_ratio', type=float, default=1.0,
                        help='the fraction of clients to update their updates during '
                             'each aggregation round')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[128])
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--lr', type=float, nargs='+', default=[0.15])
    parser.add_argument('--weight_decay', type=float, nargs="+", default=[0.0002])
    parser.add_argument('--optimizer', type=str, nargs="+", default=['sgd'])
    parser.add_argument('--dec_lr', type=int, nargs='+', default=None, help="[30, 60, 90]")
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--save_freq_epoch', type=int, default=1)


    # DP params
    parser.add_argument('--dp_option', type=str, default="None",
                        help='dpsgd for PriMIA, None for regular FL')
    parser.add_argument('--noise_multiplier', type=float, default=1.0,
                        help='ratio of Gaussian noise to add for DPSGD or bound-client training')
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="the clipping norm for Gaussian Mechanism for differential privacy"
                             "for DPSGD or bound-client. If median_clipping_norm is true and "
                             "and dp-option=bound-client, then it would be set dynamically "
                             "after each ")
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='value of delta for DP (1/num-clients ^ 1.1)')
    parser.add_argument('--target_budget', type=float, default=6.0,
                        help='the target privacy budget')
    parser.add_argument('--freeze_running_stats', type=int, default=0)

    # Dataset
    parser.add_argument('--dataset', type=str, help="pancreas,"
                                                    "xray", default="pancreas")
    parser.add_argument('--clients_to_include', type=str, default='xin_5', help='',)
    parser.add_argument('--use_keep', type=int, default=0)

    # Pancreas dataset
    parser.add_argument('--log_transform', type=int, default=1)

    # Xray dataset params
    parser.add_argument('--replace_bn_layer', type=int, default=0)
    parser.add_argument('--use_instance_norm', type=int, default=0)
    parser.add_argument('--group_norm_features', type=int, default=16)
    parser.add_argument('--drop_rate', type=float, default=0.25)

    parser.add_argument('--xray_img_size', type=int, default=224)
    parser.add_argument('--unique_patients', type=int, default=0)
    parser.add_argument('--xray_views', type=str, default='AP-PA')
    parser.add_argument('--only_include', type=str, nargs="+",
                        default=['Atelectasis', 'Effusion', 'Cardiomegaly', 'No Finding'])
    parser.add_argument('--data_aug_rot', type=int, default=15, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.05, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')

    # architecture
    parser.add_argument('--architecture', type=str, help="mnistNet, TwoNet, GEMINIModel, resnet18, etc.",
                        default="GEMINIModel")

    # exp save name
    parser.add_argument('--exp_name', type=str, default='trial')
    parser.add_argument('--exp_id', type=str, default='0')

    # Kfold cross validation
    parser.add_argument('--kfold', type=int, default=5,
                        help='kfold cv')
    # dataset path
    parser.add_argument('--dataset_path', type=str, default='dataset/Pancreas',
                        help="path to your dataset")
    parser.add_argument('--split_info_path', type=str, default='',
                        help="where to store the data split info")

    parser.add_argument('--device', type=str, default='gpu')

    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--rewrite', type=int, default=0)

    args = parser.parse_args()

    args.type_exp = 'FL'
    args.save_dir = utils.get_global_save_dir(args)
    if args.rewrite:
        print(f"rewrite dir {args.save_dir}")
        shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)

    if not os.path.exists(args.split_info_path):
        args.split_info_path = f'split_info_path/{args.dataset}'
        if not os.path.exists(args.split_info_path):
            os.makedirs(args.split_info_path)

    architecture = eval(f"custom_model.{args.architecture}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if 'xray' in args.dataset:
        args.architecture = partial(architecture, 1, len(args.only_include),
                                                     freeze_bn_layer=args.freeze_running_stats,
                                                     replace_bn_layer=args.replace_bn_layer,
                                                     groups=args.group_norm_features,
                                                     use_instance=args.use_instance_norm,
                                                     drop_rate=args.drop_rate,) # for logistic regression
    else:
        args.architecture = architecture
    if args.device != 'cpu':
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    utils.set_args_fl(args)

    args.logger = utils.get_log(args)
    args.logger.info(vars(args))

    train_fl_model(args)
