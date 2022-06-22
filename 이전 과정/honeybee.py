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
    #n_input = DatasetInput[0].shape[1]
    #n_label = DatasetLabel[0].shape[1]

    train_input, train_label, valid_input, valid_label = unpackNwindow_dataset(
        DatasetInput, DatasetLabel, IdxValid, WINDOW_SIZE)
    print("Train data shape:", train_input.shape, train_label.shape)
    print("Valid data shape:", valid_input.shape, valid_label.shape)


    ResultDir = "result/subject_%s_Honeybee/" % (to)
    if not os.path.exists(ResultDir):
        os.mkdir(ResultDir)
    save_label(dewindowing(valid_label, WINDOW_SIZE), ResultDir)

    
    #전처리@@@@@
    #train_input (N, 32, 164) -> local_train_input (N, 32, 17, 3)
    new_train_input = erasec(train_input)
    new_train_input = np.array(new_train_input)
    selected_train_input = select(new_train_input)
    npselected_train_input = np.array(selected_train_input)
    root = getroot(train_input)
    rootnp = np.array(root)
    local_train_input = local(npselected_train_input, rootnp)
    local_train_input = scaling(local_train_input)

    #valid_input (M, 32, 164) -> local_valid_input (M, 32, 17, 3)
    new_valid_input = erasec(valid_input)
    new_valid_input = np.array(new_valid_input)
    selected_valid_input = select(new_valid_input)
    npselected_valid_input = np.array(selected_valid_input)
    valid_root = getroot(valid_input)
    valid_rootnp = np.array(valid_root)
    local_valid_input = local(npselected_valid_input, valid_rootnp)
    local_valid_input = scaling(local_valid_input)

    #body_train_input, body_valid_input (N, 32, 6, 3), (M, 32, 6, 3)
    body_train_input = getbody(new_train_input)
    body_valid_input = getbody(new_valid_input)
    body_train_input = changeaxis(body_train_input)
    body_valid_input = changeaxis(body_valid_input)

    #train_label1, train_label2, valid_label1, valid_label2 (N or M, 32, 3), (N or M, 32, 279)
    train_label1 = getlabel1(train_label)
    train_label2 = getlabel2(train_label)
    valid_label1 = getlabel1(valid_label)
    valid_label2 = getlabel2(valid_label)

    local_train_input = np.array(local_train_input)
    local_valid_input = np.array(local_valid_input)
    body_train_input = np.array(body_train_input)
    body_valid_input = np.array(body_valid_input)
    train_label1 = np.array(train_label1)
    train_label2 = np.array(train_label2)
    valid_label1 = np.array(valid_label1)
    valid_label2 = np.array(valid_label2)

    #reshape
    local_train_input = local_train_input.reshape(len(local_train_input), 32, -1)
    local_valid_input = local_valid_input.reshape(len(local_valid_input), 32, -1)
    body_train_input = body_train_input.reshape(len(body_train_input), 32, -1)
    body_valid_input = body_valid_input.reshape(len(body_valid_input), 32, -1)

    print("1111Putting data to loader...", end="")
    train_dataset1 = Dataset.CMU_Dataset(
    body_train_input, train_label1, device=device)
    valid_dataset1 = Dataset.CMU_Dataset(
    body_valid_input, valid_label1, device=device)
    train_loader1 = DataLoader(dataset=train_dataset1,
                        batch_size=BATCH_SIZE, shuffle=True)
    valid_loader1 = DataLoader(dataset=valid_dataset1,
                        batch_size=BATCH_SIZE, shuffle=False)
    print("completed")

    #train_label, valid_label에서 앞에꺼 3개씩 제거하기
    print("2222Putting data to loader...", end="")
    train_dataset2 = Dataset.CMU_Dataset(
    local_train_input, train_label2, device=device)
    valid_dataset2 = Dataset.CMU_Dataset(
    local_valid_input, valid_label2, device=device)
    train_loader2 = DataLoader(dataset=train_dataset2,
                        batch_size=BATCH_SIZE, shuffle=True)
    valid_loader2 = DataLoader(dataset=valid_dataset2,
                        batch_size=BATCH_SIZE, shuffle=False)
    print("completed")

    print("Loading model/optim/scheduler...", end="")
    model1 = Network.Robust(
    n_input=18, n_output=3, n_window=WINDOW_SIZE).to(device)
    model2 = Network.Robust(
    n_input=51, n_output=279, n_window=WINDOW_SIZE).to(device)

    n_sample_train1 = train_dataset1.n_sample
    lr_step_size1 = int(n_sample_train1 / BATCH_SIZE)
    n_sample_train2 = train_dataset2.n_sample
    lr_step_size2 = int(n_sample_train2 / BATCH_SIZE)

    #loss_fn = nn.MSELoss().to(device)
    #loss_fn = nn.L1Loss().to(device)

    optimizer1 = torch.optim.AdamW(
    model1.parameters(), lr=INIT_LR, weight_decay=WEIGHT, amsgrad=True)
    lr_sch1 = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer1, step_size=lr_step_size1, gamma=0.99)

    optimizer2 = torch.optim.AdamW(
    model2.parameters(), lr=INIT_LR, weight_decay=WEIGHT, amsgrad=True)
    lr_sch2 = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer2, step_size=lr_step_size2, gamma=0.99)
    print("completed")

    print("########## Start Train ##########")
    for idx_epoch in range(EPOCH+1):
        start_time = time.time()

        train_loss1 = 0.
        train_loss2 = 0.

        for idx_batch, (x, y) in enumerate(train_loader1):
            model1.zero_grad()

            x, y = x.to(device), y.to(device)
            output1 = model1(x)

            loss1 = loss_fn1(output1, y)
            train_loss1 += loss1.item()

            #loss1.requires_grad_(True)
            loss1.backward()
            optimizer1.step()
            lr_sch1.step()

        for idx_batch, (x, y) in enumerate(train_loader2):
            model2.zero_grad()

            x, y = x.to(device), y.to(device)
            output2 = model2(x)

            loss2 = loss_fn2(output2, y)
            train_loss2 += loss2.item()

            #loss2.requires_grad_(True)
            loss2.backward()
            optimizer2.step()
            lr_sch2.step()
        

        train_loss1 /= idx_batch+1
        train_loss2 /= idx_batch+1

        valid_loss1 = 0.
        model1.eval()
        for idx_batch, (x, y) in enumerate(valid_loader1):
            x, y = x.to(device), y.to(device)
            output1 = model1(x)

            if idx_batch == 0:
                valid_pred1 = output1.cpu().data.numpy()
            else:
                valid_pred1 = np.concatenate(
                    (valid_pred1, output1.cpu().data.numpy()))
        model1.train()

        valid_loss2 = 0.
        model2.eval()
        for idx_batch, (x, y) in enumerate(valid_loader2):
            x, y = x.to(device), y.to(device)
            output2 = model2(x)

            if idx_batch == 0:
                valid_pred2 = output2.cpu().data.numpy()
            else:
                valid_pred2 = np.concatenate(
                    (valid_pred2, output2.cpu().data.numpy()))
        model2.train()

        #valid_pred = denormalization(valid_pred, label_mean, label_std)
        valid_loss1 = performance_metric(valid_pred1, valid_label1)
        valid_loss2 = performance_metric(valid_pred2, valid_label2)

        elapsed_time = time.time() - start_time

        print("result1 : \r %05d | Train Loss: %.7f | Valid Loss: %.7f | lr: %.7f | time: %.3f" % (
            idx_epoch+1, train_loss1, valid_loss1, optimizer1.param_groups[0]['lr'], elapsed_time))
        
        print("result2 : \r %05d | Train Loss: %.7f | Valid Loss: %.7f | lr: %.7f | time: %.3f" % (
            idx_epoch+1, train_loss2, valid_loss2, optimizer2.param_groups[0]['lr'], elapsed_time))


        #결과 나온거 합치기
        valid_pred = np.concatenate((valid_pred1, valid_pred2), axis = 2)
        if idx_epoch == 0 or (idx_epoch+1) % SavePeriod == 0:
            save_model1(model1, idx_epoch+1, ResultDir)
            save_model2(model2, idx_epoch+1, ResultDir)
            save_result(dewindowing(valid_pred, WINDOW_SIZE),
                        idx_epoch+1, ResultDir)





if __name__ == '__main__':
    main()

