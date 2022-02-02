import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SolveNet(nn.Module):
    def __init__(self, n_input, n_output):
        super(SolveNet, self).__init__()

        n_hidden = 256

        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        batch_size, _ = x.size()

        l1 = self.fc1(x)
        l1 = F.elu(l1)

        l2 = self.fc2(l1)
        l2 = F.elu(l2)

        l3 = self.fc3(l2)
        l3 = F.elu(l3)

        l4 = self.fc3(l3)
        l4 = F.elu(l4)
        
        l4 = self.drop(l4)

        lo = self.out(l4)

        out = lo
        return out


class SolveCNN(nn.Module):
    def __init__(self, n_input, n_output, n_window):
        super(SolveCNN, self).__init__()

        h1 = 512
        h2 = 32
        ck = 5
        pk = 2
        p = 2
        l1 = n_window + 2 * p - ck + 1
        l2 = (l1 - pk) / 2 + 1
        self.lout = int(l2)

        self.conv1 = nn.Conv1d(n_input, h1, ck, padding=p)
        self.pool1 = nn.MaxPool1d(pk, 2, return_indices=True)
        self.fc1 = nn.Linear(h1 * self.lout, h2)
        self.fc2 = nn.Linear(h2, h1 * self.lout)
        self.drop = nn.Dropout(0.5)
        self.unpool1 = nn.MaxUnpool1d(pk, 2)
        self.deconv1 = nn.ConvTranspose1d(h1, n_output, ck, padding=p)

    def forward(self, x):
        b, _, _ = x.size()
        x = torch.swapaxes(x, 1, 2)

        c1 = self.conv1(x)
        p1, i1 = self.pool1(c1)
        p1 = F.elu(p1)

        f = torch.flatten(p1, start_dim=1)

        l1 = self.fc1(f)
        l1 = F.elu(l1)

        l2 = self.fc2(l1)
        l2 = F.elu(l2)

        l2 = self.drop(l2)

        l2 = l2.reshape(b, -1, self.lout)

        u1 = self.unpool1(l2, i1)
        d1 = self.deconv1(u1)

        out = torch.swapaxes(d1, 1, 2)
        return out


class SolveRNN(nn.Module):
    def __init__(self, n_input, n_output, n_window):
        super(SolveRNN, self).__init__()

        self.w = n_window
        h = 16

        self.lstm1 = nn.LSTM(n_input, h, num_layers=2,
                             batch_first=True, dropout=0.5)
        self.out = nn.Linear(n_window * h, n_window * n_output)

    def forward(self, x):
        b, _, _ = x.size()

        r, _ = self.lstm1(x)
        r = F.elu(r)
        r = torch.flatten(r, start_dim=1)

        lo = self.out(r)

        out = lo.reshape(b, self.w, -1)
        return out

class customCNN(nn.Module):
    def __init__(self, n_input, n_output, n_window):
        super(customCNN, self).__init__()
        # L1 ImgIn shape=(32, 164, 32)
        #    Conv     -> (32, 512, 32)
        #    Pool     -> (32, 512, 16)
        
        # L2 ImgIn shape=(32, 512, 16)
        #    Conv      ->(32, 1024, 16)
        #    Pool      ->(32, 1024, 8)
        

        self.conv1 = nn.Conv1d(n_input, 512, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2,return_indices=True)
        self.conv2 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
        # Final FC 1024x8 inputs -> 282 outputs
        self.fc1 = torch.nn.Linear(1024*8, 32, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(32,1024*8,  bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.unpool1 = nn.MaxUnpool1d(2, 2)
        self.deconv1 = nn.ConvTranspose1d(512, n_output, 5, padding=2)
        self.drop = nn.Dropout(0.5)


    def forward(self, x):
        b, _, _ = x.size()
        x = torch.swapaxes(x, 1, 2)
        #print("x : ",x.shape)
        out = self.conv1(x)
        out, i1 = self.pool1(out)
        out = F.elu(out)
        
        #print("out : ",out.shape)

        out = self.conv2(out)
        out, i2= self.pool2(out)
        out = F.elu(out)
        
        #print("out : ",out.shape)

        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = F.elu(out)
        #print("out : ",out.shape)

        out = self.fc2(out)
        out = F.elu(out)
        #print("out : ",out.shape)
        out = self.drop(out)

        out = out.reshape(b, -1, 16)
        #print("out : ",out.shape)
        out = self.unpool1(out, i1)
        out = self.deconv1(out)
        #print("out : ",out.shape)
        out = torch.swapaxes(out, 1, 2)
        return out


class Robust(nn.Module):
    def __init__(self, n_input, n_output, n_window):
        super(Robust, self).__init__()
        #self.input = n_input
        #self.output = n_output
        self.dense1 = torch.nn.Linear(n_input, 2048)
        self.dense2 = torch.nn.Linear(2048, 2048)
        self.dense3 = torch.nn.Linear(2048, n_output)
        self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.dense1.weight)
        nn.init.kaiming_normal_(self.dense2.weight)
        nn.init.kaiming_normal_(self.dense3.weight)

        

    def forward(self, x):
        #print("n_input, n_output : ", self.input, self.output)
        #print("x0 : ",x.shape)
        x = self.dense1(x)
        #print("x1 : ",x.shape)

        for _ in range(5):
            save = x
            x = self.relu(x)
            x = self.dense2(x)
            x += save
            #print("x : ",x.shape, " n : ", n)

        x = self.relu(x)
        x = self.dense3(x)
        #print("x2 : ",x.shape)
       
        return x


class LSTM(nn.Module):
    def __init__(self, n_input, n_output, n_window):
        super(LSTM, self).__init__()

        self.w = n_window
        h = 16

        self.lstm1 = nn.LSTM(n_input, h, num_layers=2,
                             batch_first=True, dropout=0.2)
        self.out = nn.Linear(n_window * h, n_window * n_output)

    def forward(self, x):
        b, _, _ = x.size()

        r, _ = self.lstm1(x)
        r = F.elu(r)
        r = torch.flatten(r, start_dim=1)

        lo = self.out(r)

        out = lo.reshape(b, self.w, -1)
        return out