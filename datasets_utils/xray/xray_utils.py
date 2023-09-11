import numpy as np
import torchvision, torchvision.transforms
from sklearn.model_selection import GroupShuffleSplit


import torchxrayvision as xrv
from functools import reduce
import os

all_xray_ids = ['nih', 'pc', 'chex']


def find_useful_subset(dataset, only_include=['Atelectasis', 'Effusion', 'Cardiomegaly', 'No Finding'],
                       multiclass=False):
    ind_list = []
    if only_include:
        for label in dataset.pathologies:
            if label == 'No Finding': continue
            if label in only_include:
                ind_list.append(list(np.where(dataset.labels[:, list(dataset.pathologies).index(label)] == 1)[0]))

    if len(ind_list) > 0:
        if multiclass:
            res = reduce(np.setdiff1d, ind_list)
        else:
            res = reduce(np.union1d, ind_list)
    else:
        res = list(np.where(dataset.labels[:, list(dataset.pathologies).index("No Finding")] != 1)[0])

    try:
        no_findings = list(np.where(dataset.labels[:, list(dataset.pathologies).index("No Finding")] == 1)[0]) # add those with no findings
    except:
        no_findings = []

    return np.union1d(no_findings, res).astype(int)


def add_no_finding_to_labels(dataset):
    if "No Finding" not in dataset.pathologies:
        print('add No Finding to labels')
        no_finding = np.array(np.all(dataset.labels == 0, axis=1), dtype=float)[:, None]
        dataset.labels = np.concatenate([dataset.labels, no_finding], axis=1)
        try:
            dataset.pathologies.append("No Finding")
        except AttributeError:
            temp = dataset.pathologies.tolist()
            temp.append("No Finding")
            dataset.pathologies = temp


def get_data_augmentation(data_aug_rot=45, data_aug_trans=0.15, data_aug_scale=0.15,
                          xray_img_size=224, train=True):
    augs = []
    augs.append(xrv.datasets.ToPILImage())
    augs.append(torchvision.transforms.Resize((xray_img_size, xray_img_size)))
    if train:
        # Apply data augmentation only during training
        affine = torchvision.transforms.RandomAffine(
            data_aug_rot,
            translate=(data_aug_trans, data_aug_trans),
            scale=(1.0 - data_aug_scale, 1.0 + data_aug_scale))
        augs.append(affine)

    augs.append(torchvision.transforms.ToTensor())
    data_aug = torchvision.transforms.Compose(augs)
    return data_aug


def get_ind_save_name(dataset_name, view="", unique_patients=0):
    return f"{dataset_name}_view{view}" \
           f"_uniquePatients{unique_patients}"


def get_single_xray_data(dataset_path, dataset_name, train=True, xray_views='AP-PA',
                         xray_img_size=224, data_aug_rot=45, data_aug_trans=0.15,
                         data_aug_scale=0.15, unique_patients=0,
                         preparing=False, only_include=['No Finding']
                         ):
    if preparing:
        train = False
    views = xray_views.split("-")
    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                 ])

    data_aug = get_data_augmentation(data_aug_rot=data_aug_rot, data_aug_trans=data_aug_trans,
                                     data_aug_scale=data_aug_scale, xray_img_size=xray_img_size,
                                     train=train)

    if 'nih' in dataset_name:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=dataset_path + "/NIH/images-224",
            transform=transforms, data_aug=data_aug,
            unique_patients=unique_patients, views=views, )
    elif 'pc' in dataset_name:
        dataset = xrv.datasets.PC_Dataset(
            imgpath=dataset_path + "/padchest/images-224",
            transform=transforms, data_aug=data_aug,
            unique_patients=unique_patients, views=views, )
    elif 'chex' in dataset_name:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=dataset_path + "/chexpert/CheXpert-v1.0-small",
            csvpath=dataset_path + "/chexpert/CheXpert-v1.0-small/train.csv",
            transform=transforms, data_aug=data_aug,
            unique_patients=unique_patients, views=views, )
    else:
        raise NotImplementedError(f"{dataset_name} not supported")

    add_no_finding_to_labels(dataset)
    useful_subset = find_useful_subset(dataset, only_include)
    xrv.datasets.relabel_dataset(only_include, dataset, silent=True)
    dataset = xrv.datasets.SubsetDataset(dataset, useful_subset)
    return dataset


def prepare_xray_data(dataset_path, split_info_path, xray_views, unique_patients=0,
                      kfold=5, seed=0, recreate_data=True, only_include=["No Finding"]):

    if not os.path.exists(f"{split_info_path}/exp/fold{kfold - 1}") or recreate_data:
        for dataset_id in all_xray_ids:
            dataset = get_single_xray_data(dataset_path, dataset_id, train=False,
                                           xray_views=xray_views,
                                           unique_patients=unique_patients,
                                           preparing=True, only_include=only_include)
            print(dataset)
            if "patientid" not in dataset.csv:
                dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]
            gss = GroupShuffleSplit(train_size=(1.-1./kfold), test_size=1./kfold, random_state=seed)
            for k, (train_index, test_index) in enumerate(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid)):
                if not os.path.exists(f"{split_info_path}/exp/fold{k}"):
                    os.makedirs(f"{split_info_path}/exp/fold{k}")
                np.save(f"{split_info_path}/exp/fold{k}/{dataset_id}_ind_train.npy", train_index)
                np.save(f"{split_info_path}/exp/fold{k}/{dataset_id}_ind_test.npy", test_index)
                print(train_index.shape, test_index.shape)


def get_xray_dataset(dataset_path, split_info_path, xray_id='all', xray_views='AP-PA',
                     xray_img_size=224, data_aug_rot=45, data_aug_trans=0.15,
                     data_aug_scale=0.15, unique_patients=0,
                     train=True, exp_id=0, only_include=["No Finding"]):
    if train:
        file_name_suffix = 'train'
    else:
        file_name_suffix = 'test'

    if xray_id == 'all':
        datas = []
        for dataset_id in all_xray_ids:
            subset_ind = np.load(f"{split_info_path}/exp/fold{exp_id}/{dataset_id}_ind_{file_name_suffix}.npy")
            individual_dataset = get_single_xray_data(
                dataset_path, dataset_id, train=train, xray_views=xray_views,
                xray_img_size=xray_img_size, data_aug_rot=data_aug_rot, data_aug_trans=data_aug_trans,
                data_aug_scale=data_aug_scale, unique_patients=unique_patients,
                preparing=False, only_include=only_include,)
            train_test_subset = xrv.datasets.SubsetDataset(individual_dataset, subset_ind)
            datas.append(train_test_subset)
        dataset = xrv.datasets.Merge_Dataset(datas)

    elif xray_id in all_xray_ids:
        subset_ind = np.load(f"{split_info_path}/exp/fold{exp_id}/{xray_id}_ind_{file_name_suffix}.npy")
        individual_dataset = get_single_xray_data(
            dataset_path, xray_id, train=train, xray_views=xray_views,
            xray_img_size=xray_img_size, data_aug_rot=data_aug_rot, data_aug_trans=data_aug_trans,
            data_aug_scale=data_aug_scale, unique_patients=unique_patients, preparing=False,
            only_include=only_include,
            )
        useful_subset = xrv.datasets.SubsetDataset(individual_dataset, subset_ind)
        dataset = useful_subset

    else:
        raise NotImplementedError(f"{xray_id} not supported")
    return dataset


def get_list_private_data_xray(dataset_path, split_info_path, xray_views='AP-PA',
                     xray_img_size=224, data_aug_rot=45, data_aug_trans=0.15,
                     data_aug_scale=0.15, unique_patients=0,
                     exp_id=0, train=True, only_include=["No Finding"]):
    all_private_data = []
    for xray_id in all_xray_ids:
        all_private_data.append(get_xray_dataset(
            dataset_path, split_info_path, xray_id=xray_id, xray_views=xray_views,
            xray_img_size=xray_img_size, data_aug_rot=data_aug_rot, data_aug_trans=data_aug_trans,
            data_aug_scale=data_aug_scale, unique_patients=unique_patients,
            train=train, exp_id=exp_id, only_include=only_include))
    return all_private_data
