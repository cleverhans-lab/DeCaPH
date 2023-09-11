import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from collections import OrderedDict
import torchvision

from torch.autograd import Variable

def freeze_bn(net):
    #https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/12
    for module in net.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


class MLP_Classifier(nn.Module):
    def __init__(self, input_size=15558, output_size=4):
        super(MLP_Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class SVC(nn.Module):
    def __init__(self):
        super(SVC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(15558, 4)
        )

    def forward(self, x):
        return self.layers(x)


def densenet121(pretrained=0, num_classes=1, drop_rate=0.25,
                ):
    import torchvision
    if not pretrained:
        model = torchvision.models.densenet121(num_classes=num_classes, drop_rate=drop_rate)
        model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        model = torchvision.models.densenet121(pretrained=pretrained)
        conv0_weight = model.features.conv0.weight.clone()
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                    bias=False)
        with torch.no_grad():
            model.features.conv0.weight = nn.Parameter(
                conv0_weight.sum(dim=1, keepdim=True))
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
    model.train()
    return model
