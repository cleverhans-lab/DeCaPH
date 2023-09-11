import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.sparse import load_npz, csr_matrix
import custom_model
import random
import numpy as np
import sys
import scipy
import warnings
import copy
from typing import List


all_study_ids = ['baron_1', 'muraro_2', 'seg_3', 'wang_4', 'xin_5']


class SC_Pancreas(torch.utils.data.Dataset):
    def __init__(self, X, y, log_transform=True):
        self.features = X
        self.label = y
        self.log_transform = log_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        X = self.features[index]
        if self.log_transform:
            X.data = torch.log10(X.data + 1).to(torch.float32)
        y = self.label[index]
        return X, y


def prepare_pancreas_data(dataset_path, split_info_path, kfold=5, seed=0, recreate_data=True):
    le = preprocessing.LabelEncoder()
    target_names = ['alpha', 'beta', 'gamma', 'delta']
    le.fit(target_names)

    if recreate_data:
        data_path = f'{dataset_path}/data'
        data_list = [load_npz(os.path.join(data_path, i + '.npz')).tocsr() for i in all_study_ids]
        label_path = f"{dataset_path}/label"
        label_list = [np.load(os.path.join(label_path, i + '_label.npy'), allow_pickle=True) for i in all_study_ids]

        data_list2 = []
        label_list2 = []

        for i in range(len(data_list)):
            x = data_list[i]
            d = label_list[i]
            d = np.where((d == "alpha") | (d == "beta") | (d == 'gamma') | (d == 'delta'), d, 0)
            x = x[d != 0]
            d = d[d != 0]
            data_list2.append(x)
            label_list2.append(le.transform(d))

        for i in range(len(data_list2)):
            x, y = data_list2[i], label_list2[i]
            if not os.path.exists(f"{dataset_path}/exp/filtered/"):
                os.makedirs(f"{dataset_path}/exp/filtered/")
            scipy.sparse.save_npz(f"{dataset_path}/exp/filtered/{all_study_ids[i]}_data.npz", x)
            np.savez(f"{dataset_path}/exp/filtered/{all_study_ids[i]}_label.npz", y)
            skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            for k, (train_index, test_index) in enumerate(skf.split(np.zeros_like(y), y)):
                if not os.path.exists(f"{dataset_path}/exp/fold{k}"):
                    os.makedirs(f"{dataset_path}/exp/fold{k}")
                if not os.path.exists(f"{split_info_path}/exp/fold{k}"):
                    os.makedirs(f"{split_info_path}/exp/fold{k}")
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # save the split indices for individual studies
                np.save(f"{split_info_path}/exp/fold{k}/{all_study_ids[i]}_ind_train.npy", train_index)
                np.save(f"{split_info_path}/exp/fold{k}/{all_study_ids[i]}_ind_test.npy", test_index)

                # save the actural splitted data for individual studies
                scipy.sparse.save_npz(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_data_train.npz", x_train)
                scipy.sparse.save_npz(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_data_test.npz", x_test)
                np.savez(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_label_train.npz", y_train)
                np.savez(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_label_test.npz", y_test)

        for k in range(kfold):
            x_train_list = []
            x_test_list = []

            y_train_list = []
            y_test_list = []
            for i in range(len(all_study_ids)):
                x_train_list.append(load_npz(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_data_train.npz").tocsr())
                x_test_list.append(load_npz(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_data_test.npz").tocsr())
                y_train_list.append(np.load(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_label_train.npz")["arr_0"])
                y_test_list.append(np.load(f"{dataset_path}/exp/fold{k}/{all_study_ids[i]}_label_test.npz")["arr_0"])
            x_train_all = scipy.sparse.vstack(x_train_list)
            x_test_all = scipy.sparse.vstack(x_test_list)

            y_train_all = np.concatenate(y_train_list)
            y_test_all = np.concatenate(y_test_list)

            # save the train test for all
            scipy.sparse.save_npz(f"{dataset_path}/exp/fold{k}/all_data_train.npz", x_train_all)
            scipy.sparse.save_npz(f"{dataset_path}/exp/fold{k}/all_data_test.npz", x_test_all)
            np.savez(f"{dataset_path}/exp/fold{k}/all_label_train.npz", y_train_all)
            np.savez(f"{dataset_path}/exp/fold{k}/all_label_test.npz", y_test_all)

        # concatenate and save all data + label
        x_train_list = []
        y_train_list = []
        for i in range(len(all_study_ids)):
            x_train_list.append(load_npz(f"{dataset_path}/exp/filtered/{all_study_ids[i]}_data.npz").tocsr())
            y_train_list.append(np.load(f"{dataset_path}/exp/filtered/{all_study_ids[i]}_label.npz")["arr_0"])
        x_train_all = scipy.sparse.vstack(x_train_list)
        y_train_all = np.concatenate(y_train_list)
        scipy.sparse.save_npz(f"{dataset_path}/exp/filtered/all_data.npz", x_train_all)
        np.savez(f"{dataset_path}/exp/filtered/all_label.npz", y_train_all)


def get_torch_dataset_pancreas(X, y, log_transform=True):
    return SC_Pancreas(torch.tensor(X).to(torch.float32),
                       torch.tensor(y).to(torch.int64), log_transform)


def get_pancreas_data(dataset_path, split_info_path, study_id='all', train=True, exp_id=0, log_transform=True):
    if train:
        file_name_suffix = 'train'
    else:
        file_name_suffix = 'test'

    if study_id == 'all' or study_id in all_study_ids:
        X = load_npz(os.path.join(f"{dataset_path}/exp/fold{exp_id}/{study_id}_data_{file_name_suffix}.npz")).tocsr().toarray()
        y = np.load(f"{dataset_path}/exp/fold{exp_id}/{study_id}_label_{file_name_suffix}.npz")["arr_0"]
    else:
        raise NotImplementedError(f"the study id {study_id} is not supported")

    return get_torch_dataset_pancreas(X, y, log_transform=log_transform)


def get_list_private_data_pancreas(dataset_path, split_info_path, ids_to_include, train=True,
                                   exp_id=0,
                                   leave_one_group_out=False, group_to_test='', log_transform=True):
    all_private_data = []
    if ids_to_include == 'all':
        ids_to_include = copy.deepcopy(all_study_ids)
        if train and leave_one_group_out:
            ids_to_include.remove(group_to_test)
    elif isinstance(ids_to_include, List):
        for id in ids_to_include:
            if not id in all_study_ids:
                raise NotImplementedError(f"{id} not supported")
    else:
        raise NotImplementedError
    for study_id in ids_to_include:
        all_private_data.append(
            get_pancreas_data(dataset_path, split_info_path, study_id=study_id,
                              train=train, exp_id=exp_id,
                              log_transform=log_transform)
        )
    return all_private_data




