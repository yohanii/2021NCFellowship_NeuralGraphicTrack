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

import Dataset
import Network
from util import *


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

    ResultDir = "result/subject_%s_Run4_3/" % (to)
    if not os.path.exists(ResultDir):
        os.mkdir(ResultDir)
    save_label(dewindowing(valid_label, WINDOW_SIZE), ResultDir)

    #전처리
    """ #축바꾸기 xyzc -> zxyc
    train_input = np.array(changeaxis(train_input))
    print("Again Train data shape:", train_input.shape, train_label.shape)
    valid_input = np.array(changeaxis(valid_input))
    print("Again Valid data shape:", valid_input.shape, valid_label.shape) """

     #평균, 표준편차 구하기
    input_mean, input_std = extract_stat(train_input)
    label_mean, label_std = extract_stat(train_label)

    #정규화
    #train_input = normalization(train_input, input_mean, input_std)
    train_label = normalization(train_label, label_mean, label_std)
    #valid_input = normalization(valid_input, input_mean, input_std)
    valid_label = normalization(valid_label, label_mean, label_std) 

    train_input = np.array(newnormalization(train_input))
    print("newnormalization1 end")
    #train_label_meanstdli,train_label = newnormalization(train_label)
    #print("newnormalization2 end")
    valid_input = np.array(newnormalization(valid_input))
    print("newnormalization3 end")
    #valid_label_meanstdli,valid_label = newnormalization(valid_label)
    #print("newnormalization4 end")

    print("Putting data to loader...", end="")
    train_dataset = Dataset.CMU_Dataset(
        train_input, train_label, device=device)
    valid_dataset = Dataset.CMU_Dataset(
        valid_input, valid_label, device=device)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE, shuffle=False)
    print("completed")

    print("Loading model/optim/scheduler...", end="")
    model = Network.Robust(
        n_input=n_input, n_output=n_label, n_window=WINDOW_SIZE).to(device)

    n_sample_train = train_dataset.n_sample
    lr_step_size = int(n_sample_train / BATCH_SIZE)

    loss_fn = nn.MSELoss().to(device)

    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INIT_LR, weight_decay=WEIGHT, amsgrad=True)
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
        
        #for param in model.parameters():
        #    print(param.data)

        if idx_epoch == 0 or (idx_epoch+1) % SavePeriod == 0:
            save_model(model, idx_epoch+1, ResultDir)
            save_result(dewindowing(valid_pred, WINDOW_SIZE),
                        idx_epoch+1, ResultDir)


#change axis xyzc to zxyc
def changeaxis(input):
    new_input = []
    for i in range(len(input)):
        print(i)
        saveli1 = []
        for j in range(len(input[0])):
            if len(input[i][j])!=164:
                print("It is not 164!!")
            else:
                addr = 0
                saveli2 = []
                while addr !=164:
                    #print("before :", input[i][j][addr], input[i][j][addr+1], input[i][j][addr+2], input[i][j][addr+3])
                    saveli2.append(input[i][j][addr+2])
                    saveli2.append(input[i][j][addr])
                    saveli2.append(input[i][j][addr+1])
                    saveli2.append(input[i][j][addr+3])
                    #print("after : ", saveli2[addr], saveli2[addr+1], saveli2[addr+2], saveli2[addr+3])
                    addr+=4
                saveli1.append(saveli2)
        new_input.append(saveli1)
    print("new_input_len : ", len(new_input), len(new_input[0]), len(new_input[0][0]))
    return new_input

#get mean, std and normalization
def newnormalization(input):
    print("new normalize start!!!")
    normalized_data = []
    for i in range(len(input)):
        x = []
        y = []
        z = []
        for j in range(17):
            x.append(input[i][0][j][0])
            y.append(input[i][0][j][1])
            z.append(input[i][0][j][2])
        x_mean, x_std = extract_stat(x)
        y_mean, y_std = extract_stat(y)
        z_mean, z_std = extract_stat(z)

        saveli1 = []
        for j in range(32):
            addr = 0
            saveli2 =[]
            while addr!=164:
                saveli2.append((input[i][j][addr]-x_mean)/x_std + 1e-13)
                saveli2.append((input[i][j][addr+1]-y_mean)/y_std + 1e-13)
                saveli2.append((input[i][j][addr+2]-z_mean)/z_std + 1e-13)
                addr+=4
            saveli1.append(saveli2)
        normalized_data.append(saveli1)

    print("new normalize end!!!") 
    return normalized_data 



if __name__ == '__main__':
    main()
