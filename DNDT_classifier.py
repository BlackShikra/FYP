from functools import reduce

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.optim.lr_scheduler import ExponentialLR
from sklearn import preprocessing

device = torch.device("cuda")
data_N = pd.read_csv("modiData/bandpass_filter/N.csv")
data_V = pd.read_csv("modiData/bandpass_filter/V.csv")
data_S = pd.read_csv("modiData/bandpass_filter/S.csv")
data_F = pd.read_csv("modiData/bandpass_filter/F.csv")
data_Q = pd.read_csv("modiData/bandpass_filter/Q.csv")

# 0-N
# 1-V
# 2-S
# 3-F
# 4-Q

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
# Shuffle
data_set = data_set.sample(frac=1.0)
print(data_set)

# Read data and labels
data = pd.DataFrame(data_set.iloc[:, :260])
labels = pd.DataFrame(data_set.iloc[:, 260])
print(data)
print(labels)

# PCA
pca_train = PCA(n_components=30)
pca_train.fit(data)
data_pca = pca_train.fit_transform(data)
print(data_pca.shape)
print(pca_train.explained_variance_ratio_.sum())

# LCA
lda_train = LinearDiscriminantAnalysis(n_components=4)
lda_train.fit(data, labels)
data_lda = lda_train.transform(data)
print(data_lda.shape)
print(lda_train.explained_variance_ratio_.sum())

# Min-Max normalization
min_max_scaler_data = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(data_lda))
print(min_max_scaler_data)

# Divide training set and test set 8:2
# x is data, y is labels
split_idx = int(0.8 * len(data))
x_train = min_max_scaler_data[:split_idx]
x_test = min_max_scaler_data[split_idx:]

assert len(data) == len(x_train) + len(x_test)

split_idy = int(0.8 * len(labels))
y_train = labels[:split_idy]
y_test = labels[split_idy:]

assert len(data) == len(y_train) + len(y_test)

print(y_train['label'].value_counts())

nums = y_train['label'].value_counts().tolist()
num_N, num_V, num_S, num_F, num_Q = nums[0], nums[1], nums[2], nums[3], nums[4]
num_data = num_N + num_V + num_S + num_F + num_Q
print(num_N, num_V, num_S, num_F, num_Q, num_data)

# # 划分训练测试集
# split = int(0.75 * len(data))
#
# x_train = data[:split]
# x_test = data[split:]
# y_train = labels[:split]
# y_test = labels[split:]

_x_train = np.array(x_train)
_x_test = np.array(x_test)
_y_train = np.array(y_train)
_y_test = np.array(y_test)
d = _x_train.shape[1]

train_data = torch.from_numpy(_x_train.astype(np.float32))
train_target = torch.from_numpy(_y_train)
# train_data, train_target = train_data.to(device), train_target.to(device)

test_data = torch.from_numpy(_x_test.astype(np.float32))
test_target = torch.from_numpy(_y_test)
# x, y = x.to(device), y.to(device)

print(train_data.shape)
print(train_target.shape)
print(test_data.shape)
print(test_target.shape)
print(y_test.iloc[:, 0].tolist())


def torch_kron_prod(a, b):
    res = torch.einsum('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res


def torch_bin(x, cut_points, temperature=0.1):
    # x is an N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = cut_points.shape[0]
    W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points, _ = torch.sort(cut_points)  # make sure cut_points is monotonically increasing
    b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
    h = torch.matmul(x, W) + b
    res = torch.exp(h - torch.max(h))
    res = res / torch.sum(res, dim=-1, keepdim=True)
    return h


def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    leaf = reduce(torch_kron_prod,
                  map(lambda z: torch_bin(x[:, z[0]:z[0] + 1], z[1], temperature), enumerate(cut_points_list)))
    return torch.matmul(leaf, leaf_score)


# Cut points
num_cut = []
for i in range(3):
    num_cut.append(1)
# num_cut = [1]
num_leaf = np.prod(np.array(num_cut) + 1)
num_class = 5
cut_points_list = [torch.rand([i], requires_grad=True) for i in num_cut]
leaf_score = torch.rand([num_leaf, num_class], requires_grad=True)
loss_function = torch.nn.CrossEntropyLoss(
    weight=torch.from_numpy(np.array(
        [num_data / (5 * num_N),
         num_data / (5 * num_V),
         num_data / (5 * num_S),
         num_data / (5 * num_F),
         num_data / (5 * num_Q)]
    )).float(),
    reduction='mean'
)
optimizer = torch.optim.Adam(cut_points_list + [leaf_score], lr=0.5)
scheduler = ExponentialLR(optimizer, gamma=0.98)

# Training
for batch in range(1000):
    optimizer.zero_grad()
    y_pred = nn_decision_tree(train_data, cut_points_list, leaf_score, temperature=0.1)
    loss = loss_function(y_pred, train_target.squeeze(dim=1))
    loss.backward()
    optimizer.step()
    if batch % 100 == 0:
        print(loss.detach().numpy())
        scheduler.step()

# Test
y_pred = nn_decision_tree(test_data, cut_points_list, leaf_score, temperature=0.1)
predicted = np.argmax(y_pred.detach().numpy(), axis=1)  # The expected output of the test sample
expected = y_test.iloc[:, 0].tolist()   # Test sample prediction

# 输出结果
print(metrics.classification_report(expected, predicted))  # Output results, accuracy, recall rate, f-1 score
print(metrics.confusion_matrix(expected, predicted))  # Confusion matrix

# auc = metrics.roc_auc_score(y_test, predicted)
accuracy = metrics.accuracy_score(expected, predicted)  # Get accuracy
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('error rate %.2f' % (1 - np.mean(np.argmax(y_pred.detach().numpy(), axis=1) == y_test.iloc[:, 0].tolist())))
