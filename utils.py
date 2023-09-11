import logging
import os
import pandas as pd
import numpy as np
import torch
import shutil
from collections import defaultdict

import custom_model
from datasets_utils.pancreas.pancreas_utils import prepare_pancreas_data, \
    get_pancreas_data, get_list_private_data_pancreas, all_study_ids
from datasets_utils.xray.xray_utils import prepare_xray_data, \
    get_xray_dataset, get_list_private_data_xray, all_xray_ids
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
import pickle
from torch.utils.data import BatchSampler, Sampler
from collections import OrderedDict
from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional, Iterable
import sklearn
import torch.nn as nn
from itertools import chain
from tqdm import tqdm
import torchvision

from privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_args_fl(args):
    if "xray" in args.dataset:
        args.num_clients = len(all_xray_ids)
        print(f"num clients: {args.num_clients}")

    if args.dataset.lower() == "pancreas":
        args.num_clients = len(all_study_ids)
        print(f"num clients: {args.num_clients}")

    private_trainset_list = get_dataset(args, train=True, dataset_id='all',
                                        get_list=True)
    args.list_dataset_length = np.array(list(map(len, private_trainset_list)))
    args.client_data_split_ratio = args.list_dataset_length
    print(args.client_data_split_ratio)

    if len(args.batch_size) != args.num_clients:
        if len(args.batch_size) == 1:
            args.batch_size *= args.num_clients
        else:
            raise ValueError(f"invalid input for batch_size "
                             f"{args.batch_size}")
    if len(args.physical_batch_size) != args.num_clients:
        if len(args.physical_batch_size) == 1:
            args.physical_batch_size *= args.num_clients
        else:
            raise ValueError(f"invalid input for physical_batch_size "
                             f"{args.physical_batch_size}")

    if len(args.lr) != args.num_clients:
        if len(args.lr) == 1:
            args.lr *= args.num_clients
        else:
            raise ValueError(f"invalid input for lr "
                             f"{args.lr}")

    if len(args.optimizer) != args.num_clients:
        if len(args.optimizer) == 1:
            args.optimizer *= args.num_clients
        else:
            raise ValueError(f"invalid input for optimizer "
                             f"{args.optimizer}")

    if len(args.weight_decay) != args.num_clients:
        if len(args.weight_decay) == 1:
            args.weight_decay *= args.num_clients
        else:
            raise ValueError(f"invalid input for weight_decay "
                             f"{args.weight_decay}")

    if args.dp_option != 'None':
        args.freeze_running_stats = 1


def set_args(args):
    private_trainset = get_dataset(args, train=True, dataset_id='all',
                                   get_list=False)
    private_trainset_list = get_dataset(args, train=True, dataset_id='all', get_list=True)

    args.sampling_rate = args.batch_size / len(private_trainset)
    args.list_dataset_length = np.array(list(map(len, private_trainset_list)))

    args.list_batch_sizes = np.round(args.list_dataset_length * args.sampling_rate).astype(int)
    args.batch_size = int(sum(args.list_batch_sizes))
    sampling_rate = max(args.list_batch_sizes / args.list_dataset_length)
    args.sampling_rate = sampling_rate

    if not args.no_dp:
        args.target_delta = min(args.delta, 1 / (len(private_trainset) * 1.1))

    if not args.no_dp:
        args.freeze_running_stats = 1


def get_log(args):
    '''
    get logger
    :param args:
    :return: logger
    '''
    logger = logging.getLogger('{}-log'.format(args.dataset))
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log'))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def prepare_dataset(args):

    if args.dataset.lower() == 'pancreas':
        print("preparing pancreas data for train test split")
        prepare_pancreas_data(args.dataset_path, args.split_info_path, kfold=args.kfold,
                              seed=args.seed, recreate_data=args.recreate_data)
    elif "xray" in args.dataset:
        print("preparing xray data for train test split")
        prepare_xray_data(args.dataset_path, args.split_info_path, args.xray_views,
                          unique_patients=args.unique_patients,
                          kfold=args.kfold,
                          seed=args.seed, recreate_data=args.recreate_data,
                          only_include=args.only_include)
    else:
        raise NotImplementedError(f"split data for {args.dataset} dataset is not implemented")


def get_dataset(args, train, dataset_id='all',
                get_list=False):
    """
    get the dataset
    @param args:
    @param train:
    @param download:
    @return:
    """
    if args.dataset.lower() == 'pancreas':
        if not get_list:
            data = get_pancreas_data(args.dataset_path, args.split_info_path,
                                     study_id=dataset_id,
                                     train=train, exp_id=args.exp_id,
                                     log_transform=args.log_transform
                                     )
        else:
            data = get_list_private_data_pancreas(args.dataset_path, args.split_info_path,
                                                  ids_to_include=dataset_id,
                                                  train=train,
                                                  exp_id=args.exp_id,
                                                  log_transform=args.log_transform
                                                  )
    elif "xray" in args.dataset:
        if not get_list:
            data = get_xray_dataset(args.dataset_path, args.split_info_path, xray_id=dataset_id,
                                    xray_views=args.xray_views,
                                    xray_img_size=args.xray_img_size, data_aug_rot=args.data_aug_rot,
                                    data_aug_trans=args.data_aug_trans,
                                    data_aug_scale=args.data_aug_scale, unique_patients=args.unique_patients,
                                    train=train, exp_id=args.exp_id,
                                    only_include=args.only_include)
        else:
            data = get_list_private_data_xray(args.dataset_path, args.split_info_path,
                                              xray_views=args.xray_views,
                                              xray_img_size=args.xray_img_size,
                                              data_aug_rot=args.data_aug_rot,
                                              data_aug_trans=args.data_aug_trans,
                                              data_aug_scale=args.data_aug_scale,
                                              unique_patients=args.unique_patients,
                                              train=train,
                                              exp_id=args.exp_id,
                                              only_include=args.only_include)
    else:
        raise NotImplementedError("dataset not implemented")

    return data


def load_pretrained_state(model, args):
    if os.path.exists(args.initial_model_state):
        print(f"load pretrained {args.initial_model_state}")
        if isinstance(args.device, int):
            device = f"cuda:{args.device}"
        else:
            device = args.device
        model_state = torch.load(args.initial_model_state,
                                 map_location=device)
        try:
            model.load_state_dict(model_state['net'])
        except:
            new_state_dict = OrderedDict()
            for k, v in model_state['net'].items():
                if not 'classifier' in k:
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = v[-1:]
            model.load_state_dict(new_state_dict)
    else:
        print_and_log(args, f"initial state ({args.initial_model_state}) doesn't exist", 1)


def get_loss_func(args):
    """
    return the loss function object
    @param args:
    @return: loss function object
    """
    if args.dataset.lower() == 'pancreas':
        if args.architecture == custom_model.SVC:
            loss_func = torch.nn.MultiMarginLoss().to(args.device)

        elif args.architecture == custom_model.MLP_Classifier:
            loss_func = torch.nn.CrossEntropyLoss().to(args.device)
        else:
            raise NotImplementedError("loss function not implemented ")
    elif "xray" in args.dataset:
        loss_func = torch.nn.BCELoss().to(args.device)
    else:
        raise NotImplementedError("loss function not implemented ")

    return loss_func


def get_batch_data(data, dataset, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    if 'xray' in dataset:
        images = data["img"].to(device).contiguous()
        labels = data["lab"].float().to(device).contiguous()
    else:
        images, labels = data[0].to(device), data[1].to(device)

    return images, labels


def get_output(inputs, model, dataset, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    outputs = model(inputs)
    return outputs


def get_loss(outputs, labels, criterion, dataset, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    if 'xray' in dataset:
        loss = torch.zeros(1).to(device).float()
        for task in range(labels.shape[1]):
            task_output = outputs[:, task]
            task_target = labels[:, task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = criterion(task_output.float(), task_target.float())
                loss += task_loss
        loss = loss.sum()
    else:
        loss = criterion(outputs, labels)

    return loss


def mia_inference_and_score(dataset, model, dataloader,
                            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                            num_aug_mia=0, args=None, epoch_num=1):
    model.eval()
    with torch.no_grad():
        stats = []
        labels = []
        dataloader_t = tqdm(dataloader)
        logits_save_path = f"{args.save_dir}/logits"
        if not os.path.exists(logits_save_path):
            os.makedirs(logits_save_path)
        if num_aug_mia:
            augs = []
            affine = torchvision.transforms.RandomAffine(
                args.data_aug_rot,
                translate=(args.data_aug_trans, args.data_aug_trans),
                scale=(1.0 - args.data_aug_scale, 1.0 + args.data_aug_scale))
            augs.append(affine)
            data_aug = torchvision.transforms.Compose(augs)
        for i, data in enumerate(dataloader_t):
            outs = [] # get all augmentation for this minibatch
            xbatch, y = get_batch_data(data, dataset, device)
            labels.append(y.squeeze().cpu().numpy().astype(np.int64))
            if not num_aug_mia:
                for this_x in [xbatch]:  # no augmentation
                    logits = model(this_x).squeeze()
                    if len(logits.shape) == 1:
                        logits = torch.concat([1. - logits[:, None], logits[:, None]], dim=1)
                    outs.append(logits.detach().cpu().numpy())
            else:
                aug_pad = [data_aug(xbatch) for _ in range(num_aug_mia)]
                for this_x in [xbatch]+aug_pad:  # no augmentation
                    logits = model(this_x).squeeze()
                    if len(logits.shape) == 1:
                        logits = torch.concat([1. - logits[:, None], logits[:, None]], dim=1)
                    outs.append(logits.detach().cpu().numpy())
            stats.extend(np.array(outs).transpose((1, 0, 2)))

    opredictions = np.array(stats)[:,None,:,:]
    print("inference shape", np.array(opredictions).shape)

    labels = np.concatenate(labels)
    if dataset.lower() != 'xray':
        ## Be exceptionally careful.
        ## Numerically stable everything, as described in the paper.
        predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=3, keepdims=True)

    else:
        predictions = opredictions

    COUNT = predictions.shape[0]
    #  x num_examples x num_augmentations x logits
    y_true = predictions[np.arange(COUNT), :, :, labels[:COUNT]]
    print(y_true.shape)

    print('mean acc', np.mean(predictions[:, 0, 0, :].argmax(1) == labels[:COUNT]))

    predictions[np.arange(COUNT), :, :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=3)

    logit = (np.log(y_true.mean((1)) + 1e-45) - np.log(y_wrong.mean((1)) + 1e-45))
    print("score shape", logit.shape)
    return opredictions, logit


def get_mia_testloader(args):
    testdata = get_dataset(args, train=False, dataset_id='mia_inference_all',
                           get_list=False)
    mia_testloader = torch.utils.data.DataLoader(testdata, batch_size=args.eval_batch_size,
                                                 num_workers=args.num_workers, pin_memory=True,
                                                 shuffle=False)
    return mia_testloader


def save_mia_scores(aggregate_model, args, epoch_num, mia_testloader=None, num_aug_mia=0):
    scores_save_path = f"{args.save_dir}/scores"
    if not os.path.exists(scores_save_path):
        os.makedirs(scores_save_path)
    if os.path.exists(f"{scores_save_path}/{epoch_num}.npy"):
        print(f"{scores_save_path}/{epoch_num}.npy already exists. ")
        return
    if mia_testloader is None:
        mia_testloader = get_mia_testloader(args)
    opredictions, scores = mia_inference_and_score(args.dataset, aggregate_model, mia_testloader,
                                                         device=args.device,
                                                         num_aug_mia=num_aug_mia, args=args, epoch_num=epoch_num)

    np.save(f"{scores_save_path}/{epoch_num}.npy", scores)


def evaluate_model(dataset, model, dataloader,
                   device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    evaluate the model with dataloader
    @param model:
    @param dataloader:
    @return: the test accuracy
    """
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    if 'xray' in dataset:
        task_outputs = defaultdict(list)
        task_targets = defaultdict(list)
    else:
        task_outputs = []
        task_targets = []

    with torch.no_grad():
        dataloader_t = tqdm(dataloader)
        for i, data in enumerate(dataloader_t):
            images, labels = get_batch_data(data, dataset, device)
            model.to(device)
            outputs = get_output(images, model, dataset, device).detach()
            if 'xray' in dataset:
                for task in range(labels.shape[1]):
                    task_output = outputs[:, task]
                    task_target = labels[:, task]
                    mask = ~torch.isnan(task_target)
                    task_output = task_output[mask]
                    task_target = task_target[mask]
                    task_outputs[task].append(task_output.detach().cpu().numpy())
                    task_targets[task].append(task_target.detach().cpu().numpy())
            else:
                _, predicted = torch.max(outputs.data, 1)
                task_targets.extend(labels.cpu().tolist())
                task_outputs.extend(predicted.cpu().tolist())

    if 'xray' in dataset:
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task])) > 1:
                task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                # print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)
        task_aucs = np.asarray(task_aucs)
        auc = np.mean(task_aucs[~np.isnan(task_aucs)])

        # print(f'Avg AUC = {auc:4.4f}')
        # print(dict(zip(dataloader.dataset.pathologies, task_aucs)))

    return task_outputs, task_targets


def print_xray(args, y_val_dict, y_pred_dict, name="all", type="train", to_save=True, epsilon=None, best_alpha=None):
    predicted_dict = {}
    opt_thres_dict = {}
    tn_dict = {}
    fp_dict = {}
    fn_dict = {}
    tp_dict = {}
    ppv_dict = {}
    npv_dict = {}
    auc_dict = {}
    for pathologies in y_val_dict.keys():
        y_val = y_val_dict[pathologies]
        y_pred = y_pred_dict[pathologies]
        fpr, tpr, thres = sklearn.metrics.roc_curve(y_val, y_pred)
        auc_roc_score = sklearn.metrics.auc(fpr, tpr)
        auc_dict[pathologies] = auc_roc_score

        pente = tpr - fpr
        opt_thres = thres[np.argmax(pente)]
        opt_thres_dict[pathologies] = opt_thres
        predicted = np.array(y_pred > opt_thres, dtype=int)
        predicted_dict[pathologies] = predicted
        m = confusion_matrix(y_val, predicted)
        tn, fp, fn, tp = m.ravel()
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        tn_dict[pathologies] = tn
        fp_dict[pathologies] = fp
        fn_dict[pathologies] = fn
        tp_dict[pathologies] = tp
        ppv_dict[pathologies] = ppv
        npv_dict[pathologies] = npv
    report_dict = {}
    for pathologies in y_val_dict.keys():
        report_dict[pathologies] = {}
    for pathologies in y_val_dict.keys():
        report_dict[pathologies]["opt_thres"] = opt_thres_dict[pathologies]
        report_dict[pathologies]["tn"] = tn_dict[pathologies]
        report_dict[pathologies]["fp"] = fp_dict[pathologies]
        report_dict[pathologies]["fn"] = fn_dict[pathologies]
        report_dict[pathologies]["tp"] = tp_dict[pathologies]
        report_dict[pathologies]["ppv"] = ppv_dict[pathologies]
        report_dict[pathologies]["npv"] = npv_dict[pathologies]
        report_dict[pathologies]["auc"] = auc_dict[pathologies]
    report_dict["mean_auc"] = np.mean(list(auc_dict.values()))
    report_dict["eps"] = epsilon
    report_dict["best_alpha"] = best_alpha
    print_and_log(args, f"mean AUROC: {np.mean(list(auc_dict.values()))}", 'all' in name)
    print_and_log(args, auc_dict, 'all' in name)



    # opt_min = y_pred.min()
    # opt_max = y_pred.max()

    # ppv, recall, thres = sklearn.metrics.precision_recall_curve(y_val, y_pred)
    # auc_prc_score = sklearn.metrics.auc(ppv, recall)
    # print_and_log(args, f"auc prec-recall score: {auc_prc_score}", 1)
    # ppv80_thres_idx = np.where(ppv > 0.8)[0][0]
    # ppv80_thres = thres[ppv80_thres_idx - 1]
    #
    # predicted = np.array((y_pred > ppv80_thres), dtype=int)
    # m = confusion_matrix(y_val, predicted)
    # print_and_log(args, f"thres from prc conf matrix: \n {m}", 1)
    #
    # target_names = ['w/o patho', 'w/ patho']
    # print_and_log(args, f"\n{classification_report(y_val, predicted, target_names=target_names)}", 1)
    # report_dict = classification_report(y_val, predicted, target_names=target_names, output_dict=True)
    # if to_save:
    #     res_dir = f"{args.save_dir}/report_prc_{name}_{type}.csv"
    #     if os.path.exists(res_dir):
    #         df = pd.read_csv(res_dir)
    #     else:
    #         df = pd.DataFrame()
    #     report_dict['prc_auc'] = np.median(auc_prc_score)
    #     df = df.append({**report_dict}, ignore_index=True)
    #     df.to_csv(res_dir, index=False)



def print_f1(args, y_val, y_pred, name="all", type="train", epsilon=None, best_alpha=None):
    m = confusion_matrix(y_val, y_pred)
    print_and_log(args, f"conf matrix: \n {m}",'all' in name)

    target_names = ['alpha', 'beta', 'gamma', 'delta']
    print_and_log(args, f"\n{classification_report(y_val, y_pred, target_names=target_names)}",
                  'all' in name)

    report_dict = classification_report(y_val, y_pred, target_names=target_names, output_dict=True)
    list_f1 = [report_dict[cell_type]['f1-score'] for cell_type in target_names]
    report_dict['median_f1'] = np.median(list_f1)
    report_dict["eps"] = epsilon
    report_dict["best_alpha"] = best_alpha
    print_and_log(args, f"{type}: median f1 score: {np.median(list_f1)}", 1)


def print_and_log(args, message, force_print=0):
    args.logger.info(message)
    if args.verbose or force_print:
        print(message)


def print_save_epoch_results(args, i, train_acc, test_acc, y_val1, y_pred1, y_val2, y_pred2, name='all',
                             epsilon=None, best_alpha=None):

    if args.dataset.lower() == 'pancreas':
        if y_pred1 is not None and y_val1 is not None:
            print_and_log(args, f"train {name}: ",'all' in name)
            print_f1(args, y_val1, y_pred1, name=name, type='train',
                     epsilon=epsilon, best_alpha=best_alpha)
        print_and_log(args, f"testing {name}: ",'all' in name)
        print_f1(args, y_val2, y_pred2, name=name, type='test',
                 epsilon=epsilon, best_alpha=best_alpha)

    if args.dataset.lower() == 'xray':
        if y_pred1 is not None and y_val1 is not None:
            print_and_log(args, f"train {name}: ",'all' in name)
            print_xray(args, y_val1, y_pred1, name=name, type='train',
                       epsilon=epsilon, best_alpha=best_alpha)
        print_and_log(args, f"testing {name}: ",'all' in name)
        print_xray(args, y_val2, y_pred2, name=name, type='test',
                   epsilon=epsilon, best_alpha=best_alpha)


def create_sequences(batch_size, dataset_size, epochs, replace=False, drop_last=True):
    # create a sequence of data indices used for training
    sequence = np.concatenate([np.random.choice(dataset_size, size=dataset_size, replace=replace)
                               for i in range(epochs)])
    ind = [(j + 1) * batch_size for j in range(len(sequence) // batch_size)]
    sequence = np.split(sequence, ind)
    if not sequence[-1].tolist(): # last element is empty
        sequence = sequence[:-1]
    if len(sequence[-1]) < batch_size and drop_last:
        sequence = sequence[:-1]
    return sequence


def get_or_load_sequence(batch_size, dataset_size, total_epochs, drop_last=True,
                         ):
    if isinstance(dataset_size, int):
        sequence = create_sequences(batch_size=batch_size, dataset_size=dataset_size,
                                    epochs=total_epochs, drop_last=drop_last)
        return sequence
    elif isinstance(dataset_size, Iterable) and isinstance(batch_size, Iterable):
        offset = 0
        list_sequence = []
        resultant_sequence = []
        assert len(dataset_size) == len(batch_size)
        for i in range(len(dataset_size)):
            sequence = create_sequences(batch_size=batch_size[i],
                                        dataset_size=dataset_size[i],
                                        epochs=total_epochs, drop_last=drop_last)
            sequence = list(map(lambda x: x + offset, sequence))
            offset += dataset_size[i]
            list_sequence.append(sequence)
        for temp in zip(*list_sequence):
            resultant_sequence.append(np.concatenate(temp))

        return resultant_sequence

    elif isinstance(dataset_size, Iterable) and isinstance(batch_size, int):
        offset = 0
        list_sequence = []
        list_iter_sequence = []
        shuffle_list = []
        resultant_sequence = []
        for i in range(len(dataset_size)):
            sequence = create_sequences(batch_size=batch_size,
                                        dataset_size=dataset_size[i],
                                        epochs=total_epochs, drop_last=drop_last)
            sequence = list(map(lambda x: x + offset, sequence))
            offset += dataset_size[i]
            list_sequence.append(sequence)
            list_iter_sequence.append(iter(sequence))
            shuffle_list.extend([i] * len(sequence))

        # randomly shuffle all clients
        np.random.shuffle(shuffle_list)
        for client_id in shuffle_list:
            resultant_sequence.append(next(list_iter_sequence[client_id]))
        return resultant_sequence


def client_name_to_id(dataset, client_name):
    if "gemini" in dataset.lower():
        assert client_name in all_hospital_ids
        return all_hospital_ids.index(client_name)
    elif "pancreas" in dataset.lower():
        assert client_name in all_study_ids
        return all_study_ids.index(client_name)
    elif "xray" in dataset.lower():
        assert client_name in all_xray_ids
        return all_xray_ids.index(client_name)
    else:
        raise NotImplementedError(f"{dataset} not supported")


def save_state(model, optimizer, privacy_engine, save_dir, scheduler=None):
    state = {}
    state["net"] = model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    if privacy_engine is not None:
        state["privacy_engine"] = privacy_engine.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state,
               f"{save_dir}temp")
    shutil.move(f"{save_dir}temp",
                f"{save_dir}")
    # torch.save(state, save_dir)


def load_state(model, optimizer, privacy_engine, save_dir, device, scheduler=None):
    state = torch.load(save_dir, map_location=device)
    new_state_dict = OrderedDict()
    try:
        for k, v in state['net'].items():
            name = "module." + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(state['net'])
    # model.load_state_dict(state["net"])
    optimizer.load_state_dict(state["optimizer"])
    if privacy_engine is not None:
        privacy_engine.load_state_dict(state['privacy_engine'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])


def get_global_save_dir(args):
    if "single_client" in args.type_exp or "agg" in args.type_exp:
        base_dir = f"dpfl_baseline_save_dir_{args.dataset}_{args.type_exp}_{args.client_to_include}_dp{int(not args.no_dp)}"
    elif 'FL' in args.type_exp:
        base_dir = f'dpfl_fl_save_dir_{args.dataset}_sample{args.sample_clients_ratio}_dp{args.dp_option}'
    elif "DeCaPH" in args.type_exp:
        base_dir = f'dpfl_DeCaPH_save_dir_{args.dataset}_dp{int(not args.no_dp)}'
    else:
        raise NotImplementedError(f"type_exp {args.type_exp} not recognized")

    save_dir_suffix = f'{base_dir}_{args.exp_name}_fold{args.exp_id}'
    save_dir = f"outputs/{save_dir_suffix}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    return save_dir


def compute_epsilon(steps, sampling_probability, noise_multiplier, delta):
  """Computes epsilon value for given hyperparameters."""
  if noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  rdp = compute_rdp(
      q=sampling_probability,
      noise_multiplier=noise_multiplier,
      steps=int(steps),
      orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=delta)


class MyFixedBatchNorm(torch.nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, running_mean, running_var, weight, bias):
        super().__init__()
        self.mean = running_mean
        self.var = running_var
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, x):
        x = (x-self.mean[:, None, None]) / torch.pow(self.var[:, None, None] + 1e-05, 0.5) \
            * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class MySampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        >>> list(MySampler(iter([[1,2,3],[4,5,6,7],[8,9]])))
        [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    """

    def __init__(self, sampler) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        self.sampler = sampler

    def __iter__(self) :
        for batch in self.sampler:
            yield batch

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(list(self.sampler))  # type: ignore[arg-type]


class MySequenceSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, index: List) -> None:
        self.index = index

    def __iter__(self):
        return iter(self.index)

    def __len__(self) -> int:
        return len(np.concatenate(self.index))


from typing import List

import torch
from torch.utils.data import Sampler


class UniformWithReplacementSampler(Sampler[List[int]]):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    """

    def __init__(self, *, num_samples: int, sample_rate: float, generator=None):
        r"""
        Args:
            num_samples: number of samples to draw.
            sample_rate: probability used in sampling.
            generator: Generator used in sampling.
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __len__(self):
        return int(1 / self.sample_rate)

    def __iter__(self):
        num_batches = int(1 / self.sample_rate)
        while num_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
            )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            # print("batch size: ", len(indices))
            yield indices

            num_batches -= 1


class UniformWithReplacementSampler2(Sampler[List[int]]):
    r"""
    This sampler samples elements according to the Sampled Gaussian Mechanism.
    Each sample is selected with a probability equal to ``sample_rate``.
    """

    def __init__(self, *, num_samples: int, sample_rate: float, generator=None):
        r"""
        Args:
            num_samples: number of samples to draw.
            sample_rate: probability used in sampling.
            generator: Generator used in sampling.
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.generator = generator

        if self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    def __len__(self):
        return int(1 / self.sample_rate)

    def __iter__(self):
        num_batches = int(1 / self.sample_rate)
        while num_batches > 0:
            mask = np.random.binomial(1, self.sample_rate, self.num_samples).astype(bool)
            indices = np.arange(self.num_samples)[mask].tolist()
            # print("batch size: ", len(indices))
            yield indices

            num_batches -= 1
