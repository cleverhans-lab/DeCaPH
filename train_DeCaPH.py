import argparse
import os
import random
import shutil

import torch
from torch.utils.data import Subset

import client
import utils
from utils import get_dataset, evaluate_model, print_and_log, \
    save_state, load_state, MySampler, MySequenceSampler, get_or_load_sequence, compute_epsilon, \
    UniformWithReplacementSampler
import numpy as np
import warnings
from functools import partial
import copy
import custom_model


warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=UserWarning, append=True)


def train_ours(args, dp=1):
    """
        iteratively trains and stores the aggregate model.
        @param args:
        @return: None
        """
    private_trainset = get_dataset(args, train=True, dataset_id='all',
                                   get_list=False)

    testdata = get_dataset(args, train=False, dataset_id='all',
                                           get_list=False)
    testloader = torch.utils.data.DataLoader(testdata,
                                              batch_size=args.eval_batch_size,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)

    model = args.architecture().to(args.device)
    model.train()
    if args.freeze_running_stats: custom_model.freeze_bn(model)
    utils.load_pretrained_state(model, args)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.dec_lr is not None:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.dec_lr, gamma=args.gamma)
        else:
            scheduler = None
    else:
        raise NotImplementedError(f"{args.optimizer} not supported")

    if dp:
        from opacus import PrivacyEngine # Opacus 0.15.0 needed.
        privacy_engine = PrivacyEngine(model,
                                       sample_rate=args.sampling_rate,
                                       alphas=[1 + x / 10. for x in range(1, 100)] + list(range(12, 64)),
                                       target_delta=args.target_delta,
                                       noise_multiplier=args.noise_multiplier,
                                       max_grad_norm=args.max_grad_norm)
        privacy_engine.attach(optimizer)
    else:
        privacy_engine = None

    criterion = utils.get_loss_func(args)

    for i in range(args.num_global_server_epoch):
        print_and_log(args, f"start training for global step: {i + 1}", 1)
        # use the indices to sample the training data from each participant
        sequence = get_or_load_sequence(args.list_batch_sizes,
                                        args.list_dataset_length, 1,
                                        drop_last=True if dp else False
                                        )
        subset = torch.utils.data.Subset(private_trainset, np.concatenate(sequence))

        trainloader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size,
                                                  num_workers=args.num_workers, pin_memory=True)

        client.train_epoch(model, trainloader, criterion,
                           optimizer, privacy_engine,
                           args.dataset, args.device, freeze_running_stats=args.freeze_running_stats,
                           args=args, scheduler=scheduler)
        torch.cuda.empty_cache()

        if (i+1) % args.eval_freq == 0:
            all_y_pred, all_y_val = evaluate_model(args.dataset, model, testloader)
        else:
            all_y_pred, all_y_val = None, None
        if dp:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()
            num_steps = copy.deepcopy(privacy_engine.state_dict()["steps"])
            print_and_log(args, f"global epoch {i+1}: agg eps {epsilon} "
                                f"and best alpha {best_alpha}", 1)
            best_alpha = {"best_alpha": best_alpha, "num_steps": num_steps, "sampling_rate": args.sampling_rate,
                          "delta": args.target_delta}

            optimizer.privacy_engine.steps += 1
            next_epsilon, _ = optimizer.privacy_engine.get_privacy_spent()
            optimizer.privacy_engine.steps -= 1
        else:
            epsilon, best_alpha = None, None

        if (i+1) % args.save_freq_epoch == 0:
            save_state(model, optimizer, privacy_engine,
                       f"{args.save_dir}/model_epoch_{args.num_global_server_epoch}",
                       scheduler=scheduler)

        utils.print_save_epoch_results(args, i, None, None, None, None, all_y_val, all_y_pred,
                                       name=f"all", epsilon=epsilon, best_alpha=best_alpha)

        if dp and next_epsilon >= args.target_budget:
            args.num_global_server_epoch = i + 1
            print_and_log(args, f"budget all used up: step {args.num_global_server_epoch}", 1)
            break

    save_state(model, optimizer, privacy_engine,
               f"{args.save_dir}/model_epoch_{args.num_global_server_epoch}",
               scheduler=scheduler)
    if dp:
        num_steps = copy.deepcopy(privacy_engine.state_dict()["steps"])
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()
        print_and_log(args,
                      f"global epoch {args.num_global_server_epoch}: agg eps {epsilon} and best alpha {best_alpha}", 1)
    else:
        epsilon, best_alpha, num_steps = None, None, None

    best_alpha={"best_alpha": best_alpha, "num_steps": num_steps, "sampling_rate": args.sampling_rate}
    all_y_pred, all_y_val = evaluate_model(args.dataset, model, testloader)
    utils.print_save_epoch_results(args, args.num_global_server_epoch-1, None, None, None, None,
                                   all_y_val, all_y_pred, name=f"all", epsilon=epsilon, best_alpha=best_alpha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_model_state', type=str, default="")
    parser.add_argument('--num_workers', type=int, default=0)
    # DeCaPH parameters
    parser.add_argument('--num_global_server_epoch', type=int, default=2,
                        help='number of epochs the aggregate model get updated')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--physical_batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--lr', type=float, default=0.15)
    parser.add_argument('--weight_decay', type=float, default=0.0002)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.)
    parser.add_argument('--dec_lr', type=int, nargs='+', default=None, help="learning rate scheduler milestones"
                                                                            "e.g., 30 60 90")
    parser.add_argument('--gamma', type=float, default=0.2, help='learning rate scheduler gamma')
    parser.add_argument('--save_freq_epoch', type=int, default=1, help='freq to save the model checkpoints')
    parser.add_argument('--eval_freq', type=int, default=1, help='freq to evaluate the model')

    # DP params
    parser.add_argument('--no_dp', type=int, default=0,
                        help='whether to train with DP or not. default=0 for training with DP')
    parser.add_argument('--target_budget', type=float, default=6.0,
                        help='the target budget (eps) of DP')
    parser.add_argument('--noise_multiplier', type=float, default=1.0,
                        help='Gaussian noise to add for DP training')
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="the clipping norm for Gaussian Mechanism for DP")
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='value of delta for (eps-delta) dp')

    # Dataset
    parser.add_argument('--dataset', type=str, help="pancreas, xray", default="pancreas")

    # Pancreas dataset
    parser.add_argument('--log_transform', type=int, default=1, help='whether to apply log transform')

    # Xray dataset params
    parser.add_argument('--freeze_running_stats', type=int, default=1,
                        help='freeze the running stats in BatchNorm for DP training')
    parser.add_argument('--drop_rate', type=float, default=0.25)
    parser.add_argument('--only_include', type=str, nargs="+",
                        default=['Atelectasis', 'Effusion', 'Cardiomegaly', 'No Finding'],
                        help='the pathologies/labels to include')
    parser.add_argument('--xray_img_size', type=int, default=224)
    parser.add_argument('--unique_patients', type=int, default=0)
    parser.add_argument('--xray_views', type=str, default='AP-PA',
                        help='the views of the Xray. AP-PA for frontal views')
    parser.add_argument('--data_aug_rot', type=int, default=15, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.05, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')

    # architecture
    parser.add_argument('--architecture', type=str, help="MLP_Classifier, SVC, densenet121",
                        default="MLP_Classifier")

    # exp save name
    parser.add_argument('--exp_name', type=str, default='pancreas')
    parser.add_argument('--exp_id', type=str, default='0')

    # Kfold
    parser.add_argument('--kfold', type=int, default=5,
                        help='kfold cv')
    # dataset path
    parser.add_argument('--dataset_path', type=str, default='./data/Pancreas')
    parser.add_argument('--split_info_path', type=str, default='',
                        help="where to store the data split info")

    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--rewrite', type=int, default=0)

    args = parser.parse_args()

    args.type_exp = 'DeCaPH'
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
                                    drop_rate=args.drop_rate,)

    else:
        args.architecture = architecture
    if args.device != 'cpu':
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.set_args(args)
    args.logger = utils.get_log(args)
    args.logger.info(vars(args))

    train_ours(args, not args.no_dp)


