

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

import matplotlib.pyplot as plt
# 3차원 그래프를 그리기 위해서 from mpl_toolkits.mplot3d import Axes3D를 추가해줍니다.
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib.pyplot import imshow

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

""" ResultDir = "result/subject_%s_npy1/" % (to)
if not os.path.exists(ResultDir):
    os.mkdir(ResultDir)
save_label(dewindowing(valid_label, WINDOW_SIZE), ResultDir) """



x = []
y = []
z = []

#for i in range(len(train_input)):
#    for j in range(len(train_input[0])):
addr = 0
while addr!=164:
    x.append(train_input[0][0][addr])
    y.append(train_input[0][0][addr+1])
    z.append(train_input[0][0][addr+2])
    addr+=4


# figure 크기 설정
# fig = plt.figure()만 사용해도 됨.
fig = plt.figure(figsize=(5, 5))

# 3D axes를 만들기 위해 projection=’3d’ 키워드를 입력해줍니다.
ax = fig.gca(projection='3d')

# scatter() 함수에 준비된 x, y, z 배열 값을 입력해주고 
# 마커, 스타일 및 마커 색상 등을 설정할 수 있습니다.
# marker = 점의 형태
# s = 점의 크기
# c = 점의 색깔
ax.scatter(x,y,z, marker='o', s=15, c='darkgreen')

#plt.show()
plt.savefig('savefig_default2.png')
pil_im = Image.open('savefig_default2.png')
imshow(np.asarray(pil_im))