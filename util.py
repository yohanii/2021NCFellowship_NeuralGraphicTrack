import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


def dir_outer(path):
    dir_path_splits = path.split('\\')[:-1]
    return "/".join(dir_path_splits)


def extract_stat(dataset):
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    return mean, std


def performance_metric(p, y):
    metric = np.mean(np.sqrt(np.square(p-y)))  # MSE
    return metric


def load_dataset(InputPath, LabelPath):
    print("Loading dataset...", end="")
    DatasetInput = np.load(InputPath, allow_pickle=True)
    DatasetLabel = np.load(LabelPath, allow_pickle=True)
    print("completed")

    if DatasetInput.shape[0] != DatasetLabel.shape[0]:
        # print("Input: %d & Label: %d size not matched" %
        #       (DatasetInput.shape[0], DatasetLabel.shape[0]))
        exit()

    # print("***Input & Label Dataset Size:")
    n_sample = DatasetLabel.shape[0]
    print("n_sample:", n_sample)
    not_matched = []
    for i in range(n_sample):
        # print("%02d:" % (i+1), DatasetInput[i].shape, DatasetLabel[i].shape)
        if DatasetInput[i].shape[0] != DatasetLabel[i].shape[0]:
            # print("Input[%d]: %d & Label[%d]: %d size not matched" % (
            #     i, DatasetInput[i].shape[0], i, DatasetLabel[i].shape[0]))
            not_matched.append(i)
            # exit()

    # print("not_matched file:", len(not_matched))
    DatasetInput = np.delete(DatasetInput, not_matched)
    DatasetLabel = np.delete(DatasetLabel, not_matched)

    return DatasetInput, DatasetLabel


def unpack_dataset(DatasetInput, DatasetLabel, idx_valid):
    print("Unpacking dataset...", end="")
    valid_input = DatasetInput[idx_valid]
    valid_label = DatasetLabel[idx_valid]
    DatasetInput = np.delete(DatasetInput, idx_valid)
    DatasetLabel = np.delete(DatasetLabel, idx_valid)
    train_input = np.concatenate(DatasetInput, axis=0)
    train_label = np.concatenate(DatasetLabel, axis=0)
    print("completed")

    return train_input, train_label, valid_input, valid_label

def windowing(input_, nWindow):
    # print("Windowing dataset...", end="")
    inputs = []
    for i in range(input_.shape[0] - nWindow + 1):
        inputs.append(input_[i:i+nWindow, :])
    # print("completed")
    return np.array(inputs)


def dewindowing(inputs, nWindow):
    # print("Dewindowing dataset...", end="")
    L = inputs.shape[0]
    sum = np.pad(inputs[0], ((0, L - 1), (0, 0)), 'constant')
    for i in range(L):
        if i != 0:
            sum += np.pad(inputs[i], ((i, L - 1 - i), (0, 0)), 'constant')

    for i in range(nWindow - 1):
        sum[i] /= (i + 1)
        sum[L + nWindow - 1 - i - 1] /= (i + 1)
    sum[nWindow - 1:L] /= nWindow
    # print("completed")
    return sum

def unpackNwindow_dataset(DatasetInput, DatasetLabel, idx_valid, nWindow):
    print("Unpacking & windowing dataset...", end="")
    valid_input = windowing(DatasetInput[idx_valid], nWindow)
    valid_label = windowing(DatasetLabel[idx_valid], nWindow)
    # print(valid_input.shape, valid_label.shape)
    DatasetInput = np.delete(DatasetInput, idx_valid)
    DatasetLabel = np.delete(DatasetLabel, idx_valid)
    for i in range(DatasetInput.shape[0]):
        DatasetInput[i] = windowing(DatasetInput[i], nWindow)
        DatasetLabel[i] = windowing(DatasetLabel[i], nWindow)
    # print(DatasetInput.shape, DatasetLabel.shape)
    train_input = np.concatenate(DatasetInput, axis=0)
    train_label = np.concatenate(DatasetLabel, axis=0)
    # print(train_input.shape, train_label.shape)
    print("completed")

    return train_input, train_label, valid_input, valid_label


def normalization(data, mean, std):
    data -= mean
    data /= std + 1e-13
    return data


def denormalization(data, mean, std):
    data *= std
    data += mean
    return data


def save_model(model, step, dir):
    fname = "{:06d}_model.pt"
    torch.save(model.state_dict(), dir + fname.format(step))
    print("Model saved.")


def save_result(data, step, dir):
    data = data.astype(np.float32)
    fname = dir + "{:06d}_result_valid"
    np.save(fname.format(step), data)
    print("Result saved. ", fname.format(step))


def save_label(data, dir):
    data = data.astype(np.float32)
    fname = "valid_label"
    np.save(dir + fname, data)


def check_device():
    print("### Device Check list ###")
    print("GPU available?:", torch.cuda.is_available())
    device_number = torch.cuda.current_device()
    print("Device number:", device_number)
    print("Is device?:", torch.cuda.device(device_number))
    print("Device count?:", torch.cuda.device_count())
    print("Device name?:", torch.cuda.get_device_name(device_number))
    print("### ### ### ### ### ###\n\n")