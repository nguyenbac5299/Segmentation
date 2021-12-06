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
model = PSPNet(n_classes=21)


# loss
class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        # aux_weight : quyet dinh 2 cai loss cua 2 cai di ra trong mo hinh
        self.aux_weight = aux_weight

    def forward(self, outputs, targets):
        loss = f.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = f.cross_entropy(outputs[1], targets, reduction='mean')

        return loss + self.aux_weight * loss_aux


# loss hay de bien criterion
criterion = PSPLoss(aux_weight=0.4)

# optimizer: chien thuat update
# optimizer de update cac parameter khi training theo 1 thuat toan nao do (VD: SGD,...)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer = optim.SGD([
    {'params': model.feature_conv.parameters(), 'lr': 1e-3},
    {'params': model.feature_res_1.parameters(), 'lr': 1e-3},
    {'params': model.feature_res_2.parameters(), 'lr': 1e-3},
    {'params': model.feature_dilated_res_1.parameters(), 'lr': 1e-3},
    {'params': model.feature_dilated_res_2.parameters(), 'lr': 1e-3},
    {'params': model.pyramid_pooling.parameters(), 'lr': 1e-3},
    {'params': model.decode_feature.parameters(), 'lr': 1e-2},
    {'params': model.aux.parameters(), 'lr': 1e-2},
], momentum=0.9, weight_decay=0.001)


# scheduler
# thay doi do update
def lambda_epoch(epoch):
    max_epoch = 100
    return math.pow(1 - epoch / max_epoch, 0.9)


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)

# train model
def train_model(model, dataloader_dict, criterion, scheduler, optimizer, num_epochs):
    pass
