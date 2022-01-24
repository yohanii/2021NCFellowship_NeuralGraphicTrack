import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.backends import cudnn

import torch.utils.model_zoo as model_zoo
import torchvision.models.resnet as resnet

import Dataset
import Network
from util import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x


        out = self.conv1(x) # 3x3 stride = 2
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 stride = 1
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes) #conv1x1(64,64)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x3(planes, planes, stride)#conv3x3(64,64)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion) #conv1x1(64,256)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #t, _, _ = x.size()
        #print("x : ",x.shape," t : ", t)
        #print("In the Bottleneck")
        identity = x
        #print("x : ",x.shape)

        out = self.conv1(x) # 1x1 stride = 1
        out = self.bn1(out)
        out = self.relu(out)
        #print("x : ",x.shape)

        out = self.conv2(out) # 3x3 stride = stride 
        out = self.bn2(out)
        out = self.relu(out)
        #print("x : ",x.shape)

        out = self.conv3(out) # 1x1 stride = 1
        out = self.bn3(out)

        #print("x : ",x.shape)

        if self.downsample is not None :
            identity = self.downsample(x)
            

        out += identity
        out = self.relu(out)
        #print("x : ",x.shape)
        #print("Bottleneck end")

        return out



class ResNet(nn.Module):
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) #resnet 50 
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        self.inplanes = 64
               
        self.conv1 = nn.Conv1d(164, 64, kernel_size=5, stride=2, padding=2, bias=False)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, return_indices=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d((1))
        self.fc1 = nn.Linear(512 * block.expansion, 32)
        self.fc2 = nn.Linear(32, 512)

        self.unpool1 = nn.MaxUnpool1d(2, 2)
        self.deconv1 = nn.ConvTranspose1d(64, 282, 6,stride=2,  padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion: 
            
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), #conv1x1(256, 512, 2)
                nn.BatchNorm1d(planes * block.expansion), #batchnrom2d(512)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion #self.inplanes = 128 * 4
        
        for _ in range(1, blocks): 
            layers.append(block(self.inplanes, planes)) # * 3

        return nn.Sequential(*layers)
    
    

    def forward(self, x):
        b, _, _ = x.size()
        x = torch.swapaxes(x, 1, 2)
        #print("1x : ",x.shape) #x :  torch.Size([32, 164, 32])
        x = self.conv1(x)
        #print("2x : ",x.shape) #x :  torch.Size([32, 64, 16])
        x = self.bn1(x)
        #print("3x : ",x.shape) #x :  torch.Size([32, 64, 16])
        x = self.relu(x)
        #print("4x : ",x.shape) #x :  torch.Size([32, 64, 16])
        x, i1 = self.maxpool(x)
        #print("5x : ",x.shape, "i1 : ", i1.shape) #x :  torch.Size([32, 64, 8])

        #x = torch.swapaxes(x, 1, 2)    
        #print("x : ",x.shape)
        x = self.layer1(x)
        #print("6x : ",x.shape) #x :  torch.Size([32, 256, 8])
        x = self.layer2(x)
        #print("7x : ",x.shape) #x :  torch.Size([32, 512, 4])
        x = self.layer3(x)
        #print("8x : ",x.shape) #x :  torch.Size([32, 1024, 2])
        x = self.layer4(x)
        #print("9x : ",x.shape) #x :  torch.Size([32, 2048, 1])

        x = self.avgpool(x)
        #print("10x : ",x.shape) #x :  torch.Size([32, 2048, 1])
        x = x.view(x.size(0), -1)
        #print("11x : ",x.shape) #x :  torch.Size([32, 2048])
        x = self.fc1(x)
        #print("12x : ",x.shape) #x :  torch.Size([32, 32])

        x = self.fc2(x)
        #print("13x : ",x.shape) #x :  torch.Size([32, 512])

        #unpool, deconv해주기
        x = x.reshape(32, -1, 8)
        #print("14x : ",x.shape) #14x :  torch.Size([32, 64, 8])
        x = self.unpool1(x, i1) 
        #print("15x : ",x.shape) #15x :  torch.Size([32, 64, 16])
        x = self.deconv1(x)
        #print("16x : ",x.shape) #16x :  torch.Size([32, 282, 16])

        x = torch.swapaxes(x, 1, 2)

        return x

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs) #=> 2*(2+2+2+2) +1(conv1) +1(fc)  = 16 +2 =resnet 18
    return model
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) #=> 3*(3+4+6+3) +(conv1) +1(fc) = 48 +2 = 50
    return model
def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs) # 3*(3+8+36+3) +2 = 150+2 = resnet152    
    return model

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

    ResultDir = "result/subject_%s_ResNet/" % (to)
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
    
    res = resnet50()
    #res = resnet18()
    #res = resnet152()

    model = res.to(device)


    n_sample_train = train_dataset.n_sample
    lr_step_size = int(n_sample_train / BATCH_SIZE)


    loss_fn = nn.MSELoss().to(device)

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

            #print("shape : ", output.shape, y.shape)
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
