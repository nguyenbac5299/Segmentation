import main
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    model.to(device)
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader_dict['train'].dataset)
    num_val_imgs = len(dataloader_dict['val'].dataset)
    batch_size = dataloader_dict['train'].batch_size

    iteration = 1
    logs = []
    batch_multiplier = 3

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('Epoch {} / {} '.format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                scheduler.step()
                optimizer.zero_grad()
                print('train')
            else:
                if (epoch + 1) % 5 == 0:
                    model.eval()
                    print('val')
                else:
                    continue

            count = 0
            for images, anno_class_images in dataloader_dict[phase]:
                if images.size()[0] == 1:
                    continue

                images = images.to(device)
                anno_class_images = anno_class_images.to(device)

                if phase == 'train' and count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                with torch.set_grad_enabled(phase='train'):
                    outputs = model(images)
                    loss = criterion(outputs, anno_class_images.long()) / batch_multiplier

                    if phase == 'train':
                        loss.backward()
                        count -= 1

                        if iteration % 10 ==0:
                            t_iter_end= time.time()
                            duration= t_iter_end - t_iter_start
                            print('Iteration{} || Loss: {:.6f} || 10iter: {:.6f} sec'.format(iteration, loss.item() / batch_size * batch_multiplier, duration))
                            t_iter_start= time.time()

                        epoch_train_loss += loss.item()* batch_multiplier
                        iteration +=1
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier
        t_epoch_end= time.time()
        duration= t_epoch_end - t_epoch_start
        print('Epoch {} || Epoch_train_loss: {:.6f } || Epoch_val_loss: {:.6f}'.format(epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        print('Duration {:.6f} sec'.format(duration))
        t_epoch_start= time.time()

        torch.save(model.state_dict(),'pspnet50_' + str(epoch)+'.pth')
        

if __name__ == '__main__':
    num_epochs = 100

    root_path = 'data/VOC2012'
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_data_path_list(root_path)

    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = MyDataSet(train_img_list, train_anno_list, phase='train',
                              transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset = MyDataSet(val_img_list, val_anno_list, phase='val',
                            transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    batch_size = 12
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    data_loader_dict = {
        'train': train_data_loader,
        'val': val_data_loader
    }

    train_model(model, data_loader_dict, criterion, scheduler, optimizer, num_epochs=num_epochs)
