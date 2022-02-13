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


def save_model1(model, step, dir):
    fname = "{:06d}_model1.pt"
    torch.save(model.state_dict(), dir + fname.format(step))
    print("Model saved.")

def save_model2(model, step, dir):
    fname = "{:06d}_model2.pt"
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


#train_input : x,y,z,c 중 c 버리기
def erasec(input):
    new_train_input = []
    for i in range(len(input)):
        li32 = []
        for j in range(32):
            li41 = []
            for k in range(41):
                li41.append([input[i][j][k*4],input[i][j][k*4+1], input[i][j][k*4+2]])
            li32.append(li41)
        #print(i)
        new_train_input.append(li32)
    return new_train_input

#new_train_input(104545, 32, 41, 3)을 selected_train_input(104545, 32, 17, 3)으로 변환
def select(input):
    selected_train_input = []
    for i in range(len(input)):
        slist2 = []
        for j in range(32):
            slist = []
            slist.append((input[i][j][22] + input[i][j][25])/2) #0
            slist.append((input[i][j][3] + input[i][j][6])/2) #1
            slist.append((input[i][j][0] + input[i][j][1])/2) #2
            slist.append(input[i][j][33]) #3
            slist.append(input[i][j][38]) #4
            slist.append(input[i][j][37]) #5
            slist.append(input[i][j][14]) #6
            slist.append(input[i][j][19]) #7
            slist.append(input[i][j][18]) #8
            slist.append((input[i][j][39] + input[i][j][40])/2) #9
            slist.append((input[i][j][9] + input[i][j][28]+input[i][j][4] + input[i][j][23])/4) #10
            slist.append((input[i][j][30] + input[i][j][32])/2) #11
            slist.append(input[i][j][29]) #12
            slist.append(input[i][j][35]) #13
            slist.append((input[i][j][11] + input[i][j][13])/2) #14
            slist.append(input[i][j][10]) #15
            slist.append(input[i][j][16]) #16
            slist2.append(slist)
        selected_train_input.append(slist2)
        #print(i)
    return selected_train_input

def local(input, root):
    #상대좌표로 바꾸기 selected_train_input을
    local_train_input = np.empty((0,32,17,3))
    for i in range(len(input)):
        saveli = np.empty((0,17,3))
        for j in range(32):
            saveli = np.append(saveli, [input[i][j] - root[i][j]], axis = 0)
        #print("saveli : ",saveli.shape)
        local_train_input = np.append(local_train_input, [saveli], axis =0)
        #print(i)
    #print(local_train_input.shape)
    return local_train_input

#root 추출
def getroot(input):
    root = []
    for t in range(len(input)):
        semi = []
        for i in range(32):
            x = []
            y = []
            z = []
            x.append(input[t][i][9*4])
            y.append(input[t][i][9*4+1])
            z.append(input[t][i][9*4+2])
            x.append(input[t][i][28*4])
            y.append(input[t][i][28*4+1])
            z.append(input[t][i][28*4+2])
            x.append(input[t][i][39*4])
            y.append(input[t][i][39*4+1])
            z.append(input[t][i][39*4+2])
            x.append(input[t][i][4*4])
            y.append(input[t][i][4*4+1])
            z.append(input[t][i][4*4+2])
            x.append(input[t][i][23*4])
            y.append(input[t][i][23*4+1])
            z.append(input[t][i][23*4+2])
            x.append(input[t][i][40*4])
            y.append(input[t][i][40*4+1])
            z.append(input[t][i][40*4+2])
            semi.append([sum(x)/6, sum(y)/6, sum(z)/6])
        root.append(semi)
    return root

#train_input 스케일링2 키로 z방향이 키일 경우
def scaling(input):
    for i in range(len(input)):
        z = []
        for k in range(17):
            z.append(input[i][0][k][2])
        leng = max(z) - min(z)
        #print(i)
        #print(leng)
        input[i] = input[i]/(leng/22)
    print(input.shape)


#new_train_input(104545, 32, 41, 3)을 body_train_input(104545, 32, 6, 3)으로 변환
def getbody(input):
    body_train_input = []
    for i in range(len(input)):
        slist2 = []
        for j in range(32):
            slist = []
            slist.append(input[i][j][9]) #0
            slist.append(input[i][j][28]) #1
            slist.append(input[i][j][39]) #2
            slist.append(input[i][j][40]) #3
            slist.append(input[i][j][4]) #4
            slist.append(input[i][j][23]) #5  
            slist2.append(slist)
        body_train_input.append(slist2)
        #print(i)
    return body_train_input


#label1 분리
def getlabel1(input):
    label1 = np.empty((len(input), 32, 3))
    for i in range(len(input)):
        for j in range(32):
                for k in range(3):
                    label1[i][j][k] = input[i][j][k]
    return label1

#label2 분리
def getlabel2(input):
    label2 = np.empty((len(input), 32, 279))
    for i in range(len(input)):
        for j in range(32):
                for k in range(3, 282):
                    label2[i][j][k-3] = input[i][j][k]
    return label2

def loss_fn1(input, target):
  result1 = (abs(input-target)).sum()/input.data.nelement()
  #result2 = ((input-target)**2).sum()/input.data.nelement()
  #result = result1 + result2 * 0.01
  result = result1
  return result

def loss_fn2(input, target):
  result1 = (abs(input-target)).sum()/input.data.nelement()
  result = result1
  return result


# x, y, z -> y,z,x body_train_input
def changeaxis(input):
    for i in range(len(input)):
        for j in range(32):
                for k in range(6):
                    save = input[i][j][k][0]
                    input[i][j][k][0] = input[i][j][k][1]
                    input[i][j][k][1] = input[i][j][k][2]
                    input[i][j][k][2] = save
