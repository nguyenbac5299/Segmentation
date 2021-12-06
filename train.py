from model import PSPNet
from torch import nn as nn
import torch.nn.functional as f
from torch import optim
import math
import torch
import torch.utils.data as data
import time
from proccess_data import make_data_path_list, MyDataSet, DataTransform

# model
model= PSPNet(n_classes=21)

# loss
class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        # aux_weight : quyet dinh 2 cai loss cua 2 cai di ra trong mo hinh
        self.aux_weight= aux_weight

    def forward(self, outputs, target):



# optimizer

# scheduler

# train model

