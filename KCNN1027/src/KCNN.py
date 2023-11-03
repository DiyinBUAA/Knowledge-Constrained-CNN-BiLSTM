import argparse
import os.path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from src.config import *
from src.read_data import PanasonicData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn import Conv1d, Dropout, Linear, LSTM
import math

seed =
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def create_dataloader():
    data_reader = PanasonicData()
    X, Y = [], []
    II = []
    UU = []
    work_conditions = data_reader.drive_cycles[:training_num - 1]
    times = []
    for wc in work_conditions:
        time, ut, I, soc, battery_temp = data_reader.get_data(
            tmp, ['Drive Cycles', wc])
        x, y = split_xy(ut, I, battery_temp, soc)
        I = I[::sampling_interval]
        time = time[::sampling_interval]
        X.append(x)
        Y.append(y)
        II.extend(I)
        UU.extend(ut)
        times.extend(time)
    # x,y,
    train_x = np.concatenate(X, axis=0)
    train_y = np.concatenate(Y, axis=0)
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x)
    train_dataset = PanasonicDataset(UU, II, train_x, train_y, times)

    wc = data_reader.drive_cycles[training_num - 1]
    time, ut, I, soc, battery_temp = data_reader.get_data(
        tmp, ['Drive Cycles', wc])

    x, y = split_xy(ut, I, battery_temp, soc)
    I = I[::sampling_interval]
    time = time[::sampling_interval]
    x = scaler.transform(x)

    val_dataset = PanasonicDataset(ut, I, x, y, time)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)

    test_loaders = dict()
    work_conditions = data_reader.drive_cycles[training_num:]
    for wc in work_conditions:
        time, ut, I, soc, battery_temp = data_reader.get_data(
            tmp, ['Drive Cycles', wc])
        x, y = split_xy(ut, I, battery_temp, soc)
        I = I[::sampling_interval]
        time = time[::sampling_interval]
        x = scaler.transform(x)
        data = PanasonicDataset(ut, I, x, y, time)
        test_loaders[wc] = DataLoader(
            dataset=data, batch_size=batch_size, shuffle=True, num_workers=num_works)
    return train_loader, val_loader, test_loaders


def split_xy(voltage, current, temp, soc):
    voltage = voltage[::sampling_interval]
    current = current[::sampling_interval]
    temp = temp[::sampling_interval]
    soc = soc[::sampling_interval]
    n = len(soc)
    X, Y = [], []
    for i in range(n):
        # np.array([voltage[i],current[i],temp[i]])
        X.append(np.array([voltage[i], current[i], temp[i]]))
        Y.append(soc[i])
    return np.array(X), np.array(Y)


class EarlyStopping():
    def __init__(self):
        self.save_path = '{}/tmp_{}_timesteps_{}_loss_{}_train_{}/model'.format(model_path, tmp, timesteps, arloss,
                                                                                training_num)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.patience = patience
        self.counter = 0
        self.val_loss_min = np.Inf
        self.early_stopping = False
        self.best_score = None

    def __call__(self, val_loss, model):

        if val_loss < self.val_loss_min:
            self.counter = 0
            self.val_loss_min = val_loss
            self.save_checkpoint(model)
        else:
            self.counter += 1
        if self.counter >= patience:
            self.early_stopping = True

    def save_checkpoint(self, model):
        save_path = os.path.join(self.save_path, 'best_model.pth')
        torch.save(model, save_path)


class EarlyStopping2():
    # 用来交换误差
    def __init__(self):
        self.save_path = '{}/tmp_{}_timesteps_{}_loss_{}_train_{}/model'.format(model_path, tmp, timesteps, arloss,
                                                                                training_num)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.patience = patience
        self.counter = 0
        self.counter1 = 0
        self.counter2 = 0
        self.val_loss_min1 = np.Inf
        self.val_loss_min2 = np.Inf
        self.change = False
        self.early_stopping = False
        self.lossname = 1
        self.patience1 = patience // 2

    def __call__(self, val_loss, model):

        if self.lossname == 1:
            if val_loss < self.val_loss_min1:
                self.counter = 0
                self.counter1 = 0
                self.val_loss_min1 = val_loss
            else:
                self.counter1 += 1
                self.counter += 1
            if self.counter1 >= self.patience1:
                self.lossname = 2

        if self.lossname == 2:
            if val_loss < self.val_loss_min2:
                self.counter = 0
                self.counter2 = 0
                self.val_loss_min2 = val_loss
            else:
                self.counter2 += 1
                self.counter += 1
            if self.counter2 >= self.patience1:
                self.lossname = 1

        if self.change >= self.patience:
            self.early_stopping == True

    def save_checkpoint(self, model):
        save_path = os.path.join(self.save_path, 'best_model.pth')
        torch.save(model, save_path)


class DTNN(nn.Module):
    def __init__(self):
        super(DTNN, self).__init__()
        self.conv1 = nn.Sequential(
            Conv1d(in_channels=features, out_channels=filters,
                   kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=kernel_size)
        )
        self.dropout = Dropout(p=dropout)
        self.conv2 = nn.Sequential(
            Conv1d(in_channels=filters, out_channels=filters,
                   kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=kernel_size)
        )

        self.blstm1 = LSTM(input_size=filters, hidden_size=hidden_units,
                           bidirectional=True, batch_first=True)
        self.blstm2 = LSTM(input_size=hidden_units * 2,
                           hidden_size=hidden_units, bidirectional=True, batch_first=True)
        self.fc = Linear(hidden_units * 2, 1)

        self.fc2 = Linear(timesteps, 2)
        # self.conv2=Conv1d()

    def forward(self, x):
        # x   (batch size, features, length)
        x = self.conv1(x)  # (batch size, filters, length)
        x = self.dropout(x)
        x = self.conv2(x)  # ()
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.blstm1(x)
        x, _ = self.blstm2(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

    def initialize(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('tanh'))


class PanasonicDataset(Dataset):
    def __init__(self, voltage, current, x, y, time):
        self.voltage = voltage
        self.current = current
        self.x = x
        self.y = y
        self.time = time

    def __len__(self):
        return self.x.shape[0] // timesteps

    def __getitem__(self, i):
        x = self.x[i:i + timesteps, :].T

        y = [self.y[i + timesteps - 2], self.y[i + timesteps - 1]]
        y = torch.tensor(y)
        I = self.current[i + timesteps - 1]
        u = self.voltage[i + timesteps - 2]
        return torch.from_numpy(x), y, I, u, self.time[i + timesteps - 1] - self.time[i + timesteps - 2]

    def __len__(self):
        return self.x.shape[0] - timesteps


class MultLoss(nn.Module):
    def __init__(self):
        super(MultLoss, self).__init__()

        b = np.array(T)
        b = torch.from_numpy(b)
        self.b = b.cuda()

        c = np.array(capa_norm)
        c = torch.from_numpy(c)
        self.mse = nn.MSELoss()
        self.c = c.cuda()

        self.lambda1 = torch.from_numpy(np.array(lambda1)).cuda()
        self.lambda2 = torch.from_numpy(np.array(lambda2)).cuda()

    def forward(self, outputs, target, I, delta_t):
        soc1 = outputs[:, 0]
        soc2 = outputs[:, 1]
        delta = torch.sub(soc2, soc1)
        b = torch.div(torch.mul(I, delta_t), self.c)

        loss1 = self.mse(outputs, target)
        loss2 = self.mse(delta, b)

        loss = torch.add(torch.mul(self.lambda1, loss1),
                         torch.mul(self.lambda2, loss2))
        return loss, loss1, loss2


class MultLossCos(nn.Module):
    def __init__(self, n):
        super(MultLossCos, self).__init__()

        b = np.array(T)
        b = torch.from_numpy(b)
        self.b = b.cuda()

        c = np.array(capa_norm)
        c = torch.from_numpy(c)
        self.mse = nn.MSELoss()
        self.c = c.cuda()
        self.n = n

    def forward(self, outputs, target, I, delta_t, batch):
        soc1 = outputs[:, 0]
        soc2 = outputs[:, 1]
        delta = torch.sub(soc2, soc1)
        b = torch.div(torch.mul(I, delta_t), self.c)

        loss1 = self.mse(outputs, target)
        loss2 = (self.mse(delta, b))
        lambda1 = (math.cos(batch / self.n * math.pi) + 1) / 2
        lambda2 = 1 - lambda1
        self.lambda1 = torch.from_numpy(np.array(lambda1)).cuda()
        self.lambda2 = torch.from_numpy(np.array(lambda2)).cuda()
        loss = torch.add(torch.mul(self.lambda1, loss1),
                         torch.mul(self.lambda2, loss2))
        return loss, loss1, loss2


# myloss=nn.MSELoss()
def train_and_val(train_loader, val_loader):
    train_losses = []
    eval_losses = []
    train_losses1 = []
    train_losses2 = []
    eval_losses1 = []
    eval_losses2 = []
    early_stopping = EarlyStopping()

    model = DTNN()
    model.initialize()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    if arloss == 'MSE':
        loss_fn = nn.MSELoss()
    elif arloss == 'Mult':
        multi_loss = MultLoss()
        loss_fn = multi_loss
    elif arloss == 'Multcos':
        multi_loss = MultLossCos(len(train_loader))
        loss_fn = multi_loss
    elif arloss == 'Multchange':
        # 交换两种loss
        early_stopping = EarlyStopping2()
        loss_fn1 = nn.MSELoss()
        loss_fn2 = MultLossCos(len(train_loader))
    val_loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        print('epoch:{} starts=================================='.format(epoch + 1))
        model.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        for i, d in enumerate(train_loader):
            x, y, I, u, delta_t = d
            x = x.to(torch.float)
            y = y.to(torch.float)
            # y=y.reshape((-1,1))
            x = x.cuda()
            y = y.cuda()

            pred_y = model(x)
            if arloss == 'MSE':
                loss = loss_fn(pred_y, y)
            elif arloss == 'Mult':
                I = I.to(torch.float)
                I = I.cuda()

                delta_t = delta_t.to(torch.float)
                delta_t = delta_t.cuda()
                loss, loss1, loss2 = loss_fn(pred_y, y, I, delta_t)
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
            elif arloss == 'Multcos':
                I = I.to(torch.float)
                I = I.cuda()

                delta_t = delta_t.to(torch.float)
                delta_t = delta_t.cuda()
                loss, loss1, loss2 = loss_fn(pred_y, y, I, delta_t, i)
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
            else:
                if early_stopping.lossname == 1:
                    loss = loss_fn1(pred_y, y)
                    train_loss1 = None
                    train_loss2 = None
                else:
                    I = I.to(torch.float)
                    I = I.cuda()

                    delta_t = delta_t.to(torch.float)
                    delta_t = delta_t.cuda()
                    loss, loss1, loss2 = loss_fn2(pred_y, y, I, delta_t, i)
                    train_loss1 += loss1.item()
                    train_loss2 += loss2.item()

            train_loss += loss.item()

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print('epoch: {}, batch: {}, loss: {}'.format(
                    epoch + 1, i + 1, loss.data))

            train_losses.append(train_loss)
            if arloss != 'MSE':
                train_losses1.append(train_loss1)
                train_losses2.append(train_loss2)

        model.eval()
        eval_loss = 0
        eval_loss1 = 0
        eval_loss2 = 0
        with torch.no_grad():
            for i, d in enumerate(val_loader):
                x, y, I, u, delta_t = d
                x = x.to(torch.float)
                # y = y.reshape((-1, 1))
                x = x.cuda()
                y = y.cuda()
                y.squeeze(0)
                pred_y = model(x)
                if arloss == 'MSE' or arloss == 'Multcos' or arloss=='Mult':
                    loss = val_loss_fn(pred_y, y)
                else:
                    if early_stopping.lossname == 1:
                        loss = loss_fn1(pred_y, y)
                        eval_loss1 = None
                        eval_loss2 = None
                    else:
                        I = I.to(torch.float)
                        I = I.cuda()

                        delta_t = delta_t.to(torch.float)
                        delta_t = delta_t.cuda()
                        loss, loss1, loss2 = loss_fn2(pred_y, y, I, delta_t, i)
                        eval_loss1 += loss1.item()
                        eval_loss2 += loss2.item()

                eval_loss += loss.item()

            eval_losses.append(eval_loss)
            if arloss != "MSE":
                eval_losses1.append(eval_loss1)
                eval_losses2.append(eval_loss2)

        print('epoch:{}, train loss:{}, eval loss:{}================================'.format(epoch, train_losses[-1],
                                                                                             eval_losses[-1]))
        early_stopping(eval_loss, model)
        if early_stopping.early_stopping:
            print("end at epoch {}".format(epoch))
            break

    try:
        if arloss == 'MSE':
            loss_result = pd.DataFrame({
                'train': train_losses,
                'val': eval_losses,
            })
        else:
            train_loss_result = pd.DataFrame({
                'train': train_losses,
                'train1': train_losses1,
                'train2': train_losses2,

            })
            val_loss_result = pd.DataFrame({
                'val': eval_losses,
                'val1': eval_losses1,
                'val2': eval_losses2,
            })

            train_loss_result.to_csv(
                '{}/tmp_{}_timesteps_{}_loss_{}_train_{}/train_loss.csv'.format(model_path, tmp, timesteps, arloss,
                                                                                training_num))
            val_loss_result.to_csv(
                '{}/tmp_{}_timesteps_{}_loss_{}_train_{}/val_loss.csv'.format(model_path, tmp, timesteps, arloss,
                                                                              training_num))
    except:
        pass
    return model


def test_model(test_loader, model):
    model.eval()
    with torch.no_grad():
        for wc in test_loader.keys():
            real = []
            pred = []
            loss = []
            for i, d in enumerate(test_loader[wc]):
                x, y, I, u, t = d
                x = x.to(torch.float)
                # y = y.reshape((-1, 1))
                x = x.cuda()
                y = y.cuda()
                y.squeeze(0)
                pred_y = model(x)

                real.append(y.data[:, -1].to('cpu').numpy())
                pred.append(pred_y.data[:, -1].to('cpu').numpy())
                loss.append(
                    pred_y.data[:, -1].to('cpu').numpy() - y.data[:, -1].to('cpu').numpy())
            real = np.concatenate(real, axis=0)
            pred = np.concatenate(pred, axis=0)
            loss = np.concatenate(loss, axis=0)
            results = pd.DataFrame({'real': real,
                                    'pred': pred,
                                    'loss': loss}, index=None)
            results.to_csv(
                os.path.join(
                    '{}/tmp_{}_timesteps_{}_loss_{}_train_{}/{}_pred_results.csv'.format(model_path, tmp, timesteps,
                                                                                         arloss, training_num,
                                                                                         wc)))
            print('finish test {}'.format(wc))


parser = argparse.ArgumentParser(description='DTNN')
parser.add_argument("--tmp", type=int, default=25)
parser.add_argument("--timesteps", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--loss_function", type=str, default='MSE')
parser.add_argument("--train_dataset_num", type=int, default=4)
parser.add_argument("--model_name", type=str, default='KCCL')
parser.add_argument("--lambda1", type=float, default=0.5)
args = parser.parse_args()

tmp = args.tmp
timesteps = args.timesteps
batch_size = args.batch_size
arloss = args.loss_function
training_num = args.train_dataset_num
model_path = "output/{}/{}".format(model_name, arloss)

lambda1 = args.lambda1
lambda2 = 1 - lambda1

if not os.path.exists(model_path):
    os.makedirs(model_path)


def train_DTNN():
    train_loader, val_loader, test_loaders = create_dataloader()
    model = train_and_val(train_loader, val_loader)
    test_model(test_loaders, model)
    return model

