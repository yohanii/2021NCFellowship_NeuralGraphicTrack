import numpy as np
import time
import os
import math

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
    WINDOW_SIZE = 30
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

    ResultDir = "result/subject_%s_run5/" % (to)
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
                              batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE, shuffle=False)
    print("completed")

    print("Loading model/optim/scheduler...", end="")
    model = Network.LSTM(
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

        if idx_epoch == 0 or (idx_epoch+1) % SavePeriod == 0:
            save_model(model, idx_epoch+1, ResultDir)
            save_result(dewindowing(valid_pred, WINDOW_SIZE),
                        idx_epoch+1, ResultDir)


if __name__ == '__main__':
    main()
