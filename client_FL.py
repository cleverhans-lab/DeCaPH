import numpy as np
import os
import types
import torch
import piq
import copy
import utils
import custom_model
import shutil
import time
from collections import OrderedDict
import opacus
# import opacus_wrong
import warnings
import pprint
import client
import torch.optim as optim
import math


# seed = 0
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


class Client:
    def __init__(self, client_id, args, sequence=None):
        self.client_id = client_id
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.batch_size = args.batch_size[client_id]
        self.physical_batch_size = args.physical_batch_size[client_id]
        self.freeze_running_stats = args.freeze_running_stats
        self.n_accumulation_steps = 1

        self.lr = args.lr[client_id]
        self.weight_decay = args.weight_decay[client_id]

        self.seed = args.seed

        self.dp = args.dp_option.lower() == 'dpsgd'
        self.noise_multiplier = args.noise_multiplier
        self.max_grad_norm = args.max_grad_norm

        self.net = args.architecture()
        self.net.to(self.device)
        self.dec_lr = args.dec_lr
        self.gamma = args.gamma
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
        if self.dec_lr is not None:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                           milestones=self.dec_lr,
                                                           gamma=self.gamma)
        else:
            self.scheduler = None
        _, self.testloader, self.private_trainset, self.testset = \
            client.get_private_trainloader_and_public_testloader(args, client_id)
        if self.dp:
            from opacus import PrivacyEngine  # Opacus 0.15.0 needed.
            self.net.train()
            self.target_budget = args.target_budget
            self.target_delta = min(args.delta, 1 / (len(self.private_trainset) * 1.1))
            self.privacy_engine = PrivacyEngine(self.net, batch_size=min(self.batch_size, self.private_trainset.__len__()),
                                           sample_size=self.private_trainset.__len__(),
                                           alphas=[1 + x / 10. for x in range(1, 100)] + list(range(12, 64)),
                                           # alphas is the orders for renyi DP
                                           noise_multiplier=self.noise_multiplier,
                                           max_grad_norm=self.max_grad_norm,
                                           target_delta=self.target_delta)  # max_grad_norm can be changed.
            self.privacy_engine.attach(self.optimizer)
        else:
            self.privacy_engine = None
        self.criterion = client.get_loss_func(args)
        # self.get_private_trainloader(sequence=sequence)

    def load_aggregate_state(self, aggregate_state):
        if isinstance(aggregate_state, str):
            state = torch.load(aggregate_state, map_location=self.device)
            self.net.load_state_dict(state)
        elif isinstance(aggregate_state, OrderedDict):
            self.net.load_state_dict(copy.deepcopy(aggregate_state))
        elif isinstance(aggregate_state, torch.nn.Module):
            self.net.load_state_dict(copy.deepcopy(aggregate_state.state_dict()))
        elif isinstance(aggregate_state, dict):
            self.net.load_state_dict(copy.deepcopy(aggregate_state["net"]))
        else:
            raise NotImplementedError("aggregate_state type not recognized")

    def get_private_trainloader(self, from_global_epoch=0, sequence=None,):
        if sequence is None:
            sequence = utils.get_or_load_sequence(self.batch_size, self.private_trainset.__len__(),
                                                  self.args.num_local_client_epoch,
                                                  f"{self.args.save_dir}/clients/g{from_global_epoch}/c{self.client_id}/",
                                                  drop_last=self.dp)
        try:
            sequence = np.concatenate(sequence)
        except ValueError:
            pass
        subset = torch.utils.data.Subset(self.private_trainset, sequence)
        self.train_loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size,
                                                  num_workers=self.args.num_workers, pin_memory=True)

    def save(self,):
        net_state_dict = self.net.state_dict()

        state = {'net': net_state_dict,
                 'optimizer': self.optimizer.state_dict(),
                 }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        if self.privacy_engine is not None:
            state["privacy_engine"] = self.privacy_engine.state_dict()
            state["eps"], state["best_alpha"] = self.privacy_engine.get_privacy_spent()
            state["delta"] = self.target_delta

        return copy.deepcopy(state)


    def train_step(self, batch_idx, data):
        inputs, labels = utils.get_batch_data(data, self.dataset, self.device)
        if not self.dp:
            outputs = utils.get_output(inputs, self.net, self.dataset, self.device)
            loss = utils.get_loss(outputs, labels, self.criterion, self.dataset, self.device)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            for num_pseudo_batch in range(math.ceil(len(inputs)/self.physical_batch_size)):
                torch.cuda.empty_cache()
                start_ind = num_pseudo_batch * self.physical_batch_size
                end_ind = (num_pseudo_batch + 1) * self.physical_batch_size
                input_temp = inputs[start_ind: end_ind].to(self.device)
                label_temp = labels[start_ind: end_ind].to(self.device)
                outputs = utils.get_output(input_temp, self.net, self.dataset, self.device)
                loss = utils.get_loss(outputs, label_temp, self.criterion, self.dataset, self.device)
                loss.backward()
                if (num_pseudo_batch + 1) == math.ceil(len(inputs)/self.physical_batch_size):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # print('step')
                else:
                    self.optimizer.virtual_step()

    def train(self, epoch, ):
        self.net.to(self.device)
        self.net.train()
        if self.freeze_running_stats:
            custom_model.freeze_bn(self.net)
        self.optimizer.zero_grad()
        for batch_idx, data in enumerate(self.train_loader, 0):
            if self.dp:
                self.optimizer.privacy_engine.steps += 1
                next_epsilon, _ = self.optimizer.privacy_engine.get_privacy_spent()
                self.optimizer.privacy_engine.steps -= 1
                if next_epsilon >= self.target_budget:
                    print("privacy budget would exceed if train for another step. break")
                    break
            self.train_step(batch_idx, data)
        if self.scheduler is not None:
            self.scheduler.step()
        if self.dp:
            epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent()
            print(f"clipping norm {self.max_grad_norm} || "
                  f"noise {self.noise_multiplier} || "
                  f"eps {epsilon} and best alpha {best_alpha} || "
                  f"steps: {self.optimizer.privacy_engine.state_dict()['steps']}")
            torch.cuda.empty_cache()

    def validate(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = utils.get_batch_data(data, self.dataset, self.device)
                outputs = utils.get_output(inputs, self.net, self.dataset, self.device)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total} %')
        return correct / total

    def report_train_acc(self):
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.predict(inputs.contiguous())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Test Accuracy: {100 * correct / total} %')
        return correct / total

    def get_batch_data(self):
        if self.batches_remaining > 0:
            data = next(self.train_loader_iter)
            self.batches_remaining -= 1
        else:
            data = None
        return data
