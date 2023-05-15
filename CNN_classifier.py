import os
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# from utils import load_data, plot_history_torch, plot_heat_map

# project root path
project_path = "./"
# define log directory
# must be a subdirectory of the directory specified when starting the web application
# it is recommended to use the date time as the subdirectory name
log_dir = project_path + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = project_path + "ecg_model.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_N = pd.read_csv("modiData/bandpass_filter/N.csv")
data_V = pd.read_csv("modiData/bandpass_filter/V.csv")
data_S = pd.read_csv("modiData/bandpass_filter/S.csv")
data_F = pd.read_csv("modiData/bandpass_filter/F.csv")
data_Q = pd.read_csv("modiData/bandpass_filter/Q.csv")

time = []
for n in range(0, 260):
    time.append(n)
label = 0
for data in [data_N, data_V, data_S, data_F, data_Q]:
    data.loc[-1] = data.columns.tolist()
    data.index = data.index + 1
    data.sort_index(inplace=True)
    data.columns = time
    data['label'] = label
    label = label + 1

data_set = pd.concat([data_N, data_V, data_S, data_F, data_Q])
print(data_set)
# 打乱顺序
data_set = data_set.sample(frac=1.0)
print(data_set)

# 读取数据和标签
data = pd.DataFrame(data_set.iloc[:, :259])
labels = pd.DataFrame(data_set.iloc[:, 260])
print(data)
print(labels)

# 划分训练集、测试集、验证集8:2
# x指数据、y指标签
split_idx = int(0.8 * len(data))
x_train = data[:split_idx]
x_test = data[split_idx:]

assert len(data) == len(x_train) + len(x_test)

split_idy = int(0.8 * len(labels))
y_train = labels[:split_idy]
y_test = labels[split_idy:]

assert len(data) == len(y_train) + len(y_test)


# define the dataset class
class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.x)


# build the CNN model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # the first convolution layer, 4 21x1 convolution kernels, output shape (batch_size, 4, 300)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding='same')
        # the first pooling layer, max pooling, pooling size=3 , stride=2, output shape (batch_size, 4, 150)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # the second convolution layer, 16 23x1 convolution kernels, output shape (batch_size, 16, 150)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding='same')
        # the second pooling layer, max pooling, pooling size=3, stride=2, output shape (batch_size, 16, 75)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # the third convolution layer, 32 25x1 convolution kernels, output shape (batch_size, 32, 75)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding='same')
        # the third pooling layer, average pooling, pooling size=3, stride=2, output shape (batch_size, 32, 38)
        self.pool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        # the fourth convolution layer, 64 27x1 convolution kernels, output shape (batch_size, 64, 38)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding='same')
        # flatten layer, for the next fully connected layer, output shape (batch_size, 38*64)
        self.flatten = nn.Flatten()
        # fully connected layer, 128 nodes, output shape (batch_size, 128)
        self.fc1 = nn.Linear(64 * 38, 128)
        # Dropout layer, dropout rate = 0.2
        self.dropout = nn.Dropout(0.2)
        # fully connected layer, 5 nodes (number of classes), output shape (batch_size, 5)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        # x.shape = (batch_size, 300)
        # reshape the tensor with shape (batch_size, 300) to (batch_size, 1, 300)
        x = x.reshape(-1, 1, 300)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# define the training function and validation function
def train_steps(loop, model, criterion, optimizer):
    train_loss = []
    train_acc = []
    model.train()
    for step_index, (X, y) in loop:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)
        pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(y, pred_result)
        train_acc.append(acc)
        loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(train_loss),
            "acc": np.mean(train_acc)}


def test_steps(loop, model, criterion):
    test_loss = []
    test_acc = []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y).item()

            test_loss.append(loss)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            acc = accuracy_score(y, pred_result)
            test_acc.append(acc)
            loop.set_postfix(loss=loss, acc=acc)
    return {"loss": np.mean(test_loss),
            "acc": np.mean(test_acc)}


def train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer):
    num_epochs = config['num_epochs']
    train_loss_ls = []
    train_loss_acc = []
    test_loss_ls = []
    test_loss_acc = []
    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        train_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')
        test_loop.set_description(f'Epoch [{epoch + 1}/{num_epochs}]')

        train_metrix = train_steps(train_loop, model, criterion, optimizer)
        test_metrix = test_steps(test_loop, model, criterion)

        train_loss_ls.append(train_metrix['loss'])
        train_loss_acc.append(train_metrix['acc'])
        test_loss_ls.append(test_metrix['loss'])
        test_loss_acc.append(test_metrix['acc'])

        print(f'Epoch {epoch + 1}: '
              f'train loss: {train_metrix["loss"]}; '
              f'train acc: {train_metrix["acc"]}; ')
        print(f'Epoch {epoch + 1}: '
              f'test loss: {test_metrix["loss"]}; '
              f'test acc: {test_metrix["acc"]}')

        writer.add_scalar('train/loss', train_metrix['loss'], epoch)
        writer.add_scalar('train/accuracy', train_metrix['acc'], epoch)
        writer.add_scalar('validation/loss', test_metrix['loss'], epoch)
        writer.add_scalar('validation/accuracy', test_metrix['acc'], epoch)

    return {'train_loss': train_loss_ls,
            'train_acc': train_loss_acc,
            'test_loss': test_loss_ls,
            'test_acc': test_loss_acc}

def main():
    config = {
        'seed': 42,  # the random seed
        'test_ratio': 0.3,  # the ratio of the test set
        'num_epochs': 30,
        'batch_size': 128,
        'lr': 0.001,
    }

    # X_train,y_train is the training set
    # X_test,y_test is the test set
    train_dataset, test_dataset = ECGDataset(x_train, y_train), ECGDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # define the model
    model = Model()
    if os.path.exists(model_path):
        # import the pre-trained model if it exists
        print('Import the pre-trained model, skip the training process')
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        # build the CNN model
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        # print the model structure
        summary(model, (config['batch_size'], x_train.shape[1]), col_names=["input_size", "kernel_size", "output_size"],
                verbose=2)

        # define the Tensorboard SummaryWriter
        writer = SummaryWriter(log_dir=log_dir)
        # train and evaluate model
        history = train_epochs(train_dataloader, test_dataloader, model, criterion, optimizer, config, writer)
        writer.close()
        # save the model
        torch.save(model.state_dict(), model_path)
        # plot the training history

    # predict the class of test data
    y_pred = []
    model.eval()
    with torch.no_grad():
        for step_index, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_result = torch.argmax(pred, dim=1).detach().cpu().numpy()
            y_pred.extend(pred_result)
    # plot confusion matrix heat map
    predicted = y_pred
    expected = y_test

    # 输出结果
    print(metrics.classification_report(expected, predicted))  # 输出结果，精确度、召回率、f-1分数
    print(metrics.confusion_matrix(expected, predicted))  # 混淆矩阵

    # auc = metrics.roc_auc_score(y_test, predicted)
    accuracy = metrics.accuracy_score(expected, predicted)  # 求精度
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == '__main__':
    main()