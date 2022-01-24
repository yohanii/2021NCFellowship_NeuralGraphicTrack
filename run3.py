import numpy as np
import time
import os
from numpy.core.numeric import indices

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn

import torch.utils.model_zoo as model_zoo

import Dataset
import Network
from util import *

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, features2, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        
        self.features = features #convolution
        self.features2 = features2 #deconvolution

        self.conv1 = nn.Conv1d(164, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2 , return_indices=True)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(512)

        self.unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose1d(512, 512, 3,stride=1,  padding=1)
        self.deconv2 = nn.ConvTranspose1d(512, 256, 3,stride=1,  padding=1)
        self.deconv3 = nn.ConvTranspose1d(256, 256, 3,stride=1,  padding=1)
        self.deconv4 = nn.ConvTranspose1d(256, 128, 3,stride=1,  padding=1)
        self.deconv5 = nn.ConvTranspose1d(128, 64, 3,stride=1,  padding=1)
        self.deconv6 = nn.ConvTranspose1d(64, 282, 3,stride=1,  padding=1)
        


        self.avgpool = nn.AdaptiveAvgPool1d((1))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512 * 1),
        )#FC layer
        


        if init_weights:
            self._initialize_weights()

        

    def forward(self, x):
        indices_list = []
        x = torch.swapaxes(x, 1, 2)
        #print("0x : ",x.shape)

        #Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        indices_list += [indices]
        #print("indice shape1 :", indices.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        indices_list += [indices]
        #print("indice shape2:", indices.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        indices_list += [indices]
        #print("indice shape3 :", indices.shape)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        indices_list += [indices]
        #print("indice shape4 :", indices.shape)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        indices_list += [indices]
        #print("indice shape5 :", indices.shape)
        #print("1x : ",x.shape)

        x = self.avgpool(x) # avgpool
        #print("2x : ",x.shape)
        x = x.view(x.size(0), -1) #view
        #print("3x : ",x.shape)
        x = self.classifier(x) #FC layer
        #print("4x : ",x.shape)

        x = x.reshape(32, -1, 1)
        #print("5x : ",x.shape)
        #deconv
        x = self.unpool(x, indices_list[-1])
        x = self.deconv1(x)
        #print("6-1x : ",x.shape)
        x = self.deconv1(x)
        #print("6-2x : ",x.shape)
        #print("indice shape :", indices_list[-2].shape)
        x = self.unpool(x, indices_list[-2])
        x = self.deconv1(x)
        #print("6-3x : ",x.shape)
        x = self.deconv2(x)
        #print("6-4x : ",x.shape)
        x = self.unpool(x, indices_list[-3])
        x = self.deconv3(x)
        #print("6-5x : ",x.shape)
        x = self.deconv4(x)
        #print("6-6x : ",x.shape)
        x = self.unpool(x, indices_list[-4])
        x = self.deconv5(x)
        #print("6-7x : ",x.shape)
        x = self.unpool(x, indices_list[-5])
        x = self.deconv6(x)

        #print("6x : ",x.shape)
        x = torch.swapaxes(x, 1, 2)
        #print("7x : ",x.shape)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False): #'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #8 + 3 =11 == vgg11
    layers = []
    in_channels = 164 #n_input
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2 , return_indices=True)]
        else:
            Conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [Conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [Conv1d, nn.ReLU(inplace=True)]
            in_channels = v
            
                     
    return nn.Sequential(*layers)


def make_layers2(cfg, batch_norm=False): #'A': ['M', 512, 512, 'M', 512, 512, 'M', 256, 256, 'M', 128, 'M', 64], #8 + 3 =11 == vgg11
    layers = []
    in_channels = 512 #n_input
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxUnpool1d(kernel_size=2, stride=2)]
        else:
            Conv1d = nn.ConvTranspose1d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [Conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [Conv1d, nn.ReLU(inplace=True)]
            in_channels = v
            
    layers += [nn.ConvTranspose1d(in_channels, 282, kernel_size=3, padding=1)]             
    return nn.Sequential(*layers)




def main():
    EPOCH = 100
    BATCH_SIZE = 32
    WINDOW_SIZE = 32
    INIT_LR = 0.001
    WEIGHT = 0.0001

    SavePeriod = 10
    IdxValid = 0

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    check_device()

    to = input("사용할 subject의 수를 입력해주세요: ")
    InputPath = "dataset/subject_%s_input_data.npy" % (to)
    LabelPath = "dataset/subject_%s_label_data.npy" % (to)

    DatasetInput, DatasetLabel = load_dataset(InputPath, LabelPath)
    n_input = DatasetInput[0].shape[1]
    n_label = DatasetLabel[0].shape[1]

    train_input, train_label, valid_input, valid_label = unpackNwindow_dataset(
        DatasetInput, DatasetLabel, IdxValid, WINDOW_SIZE)
    print("Train data shape:", train_input.shape, train_label.shape)
    print("Valid data shape:", valid_input.shape, valid_label.shape)

    ResultDir = "result/subject_%s_VGG/" % (to)
    if not os.path.exists(ResultDir):
        os.mkdir(ResultDir)
    save_label(dewindowing(valid_label, WINDOW_SIZE), ResultDir)

    input_mean, input_std = extract_stat(train_input)
    label_mean, label_std = extract_stat(train_label)

    train_input = normalization(train_input, input_mean, input_std)
    train_label = normalization(train_label, label_mean, label_std)
    valid_input = normalization(valid_input, input_mean, input_std)
    valid_label = normalization(valid_label, label_mean, label_std)

    print("Putting data to loader...", end="")
    train_dataset = Dataset.CMU_Dataset(
        train_input, train_label, device=device)
    valid_dataset = Dataset.CMU_Dataset(
        valid_input, valid_label, device=device)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE, shuffle=False, drop_last = True)
    print("completed")

    print("Loading model/optim/scheduler...", end="")
    
    cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #8 + 3 =11 == vgg11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 10 + 3 = vgg 13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], #13 + 3 = vgg 16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # 16 +3 =vgg 19
    'custom' : [64,64,64,'M',128,128,128,'M',256,256,256,'M']
    }

    cfg2 = {
        'A': ['M', 512, 512, 'M', 512, 512, 'M', 256, 256, 'M', 128, 'M', 64], #8 + 3 =11 == vgg11
    }

    conv = make_layers(cfg['A'], batch_norm=True)
    deconv = make_layers2(cfg2['A'], batch_norm=False)
    vgg = VGG(conv, deconv, num_classes=10, init_weights=True)
    
    model = vgg.to(device)

    n_sample_train = train_dataset.n_sample
    lr_step_size = int(n_sample_train / BATCH_SIZE)

    #loss_fn = nn.CrossEntropyLoss().to(device)
    loss_fn =nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=INIT_LR, weight_decay=WEIGHT)
    lr_sch = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=lr_step_size, gamma=0.99)
    print("completed")

    print("########## Start Train ##########")
    for idx_epoch in range(EPOCH+1):
        start_time = time.time()

        train_loss = 0.

        for idx_batch, (x, y) in enumerate(train_loader):
            model.zero_grad()

            x, y = x.to(device), y.to(device)
            output = model(x)

            loss = loss_fn(output, y)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_sch.step()

        train_loss /= idx_batch+1

        valid_loss = 0.
        model.eval()
        for idx_batch, (x, y) in enumerate(valid_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)

            if idx_batch == 0:
                valid_pred = output.cpu().data.numpy()
            else:
                valid_pred = np.concatenate(
                    (valid_pred, output.cpu().data.numpy()))
        model.train()

        valid_pred = denormalization(valid_pred, label_mean, label_std)
        valid_loss = performance_metric(valid_pred, valid_label)

        elapsed_time = time.time() - start_time

        print("\r %05d | Train Loss: %.7f | Valid Loss: %.7f | lr: %.7f | time: %.3f" % (
            idx_epoch+1, train_loss, valid_loss, optimizer.param_groups[0]['lr'], elapsed_time))

        if idx_epoch == 0 or (idx_epoch+1) % SavePeriod == 0:
            save_model(model, idx_epoch+1, ResultDir)
            save_result(dewindowing(valid_pred, WINDOW_SIZE),
                        idx_epoch+1, ResultDir)


if __name__ == '__main__':
    main()
