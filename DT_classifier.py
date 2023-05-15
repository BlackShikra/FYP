import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 导入DecisionTreeClassifier函数
from sklearn.decomposition import PCA

data_N = pd.read_csv("modiData/bandpass_filter/N.csv")
data_V = pd.read_csv("modiData/bandpass_filter/V.csv")
data_S = pd.read_csv("modiData/bandpass_filter/S.csv")
data_F = pd.read_csv("modiData/bandpass_filter/F.csv")
data_Q = pd.read_csv("modiData/bandpass_filter/Q.csv")

time = []
for n in range(0, 260):
    time.append(n)
label = 0
for data in [data_N, data_V, data_S, data_F]:
    data.loc[-1] = data.columns.tolist()
    data.index = data.index + 1
    data.sort_index(inplace=True)
    data.columns = time
    data['label'] = label
    label = label + 1

data_set = pd.concat([data_N, data_V, data_S, data_F])
print(data_set)
# Shuffle
data_set = data_set.sample(frac=1.0)
print(data_set)

# Read data and labels
data = pd.DataFrame(data_set.iloc[:, :260])
labels = pd.DataFrame(data_set.iloc[:, 260])
print(data)
print(labels)

# # PCA
# pca_train = PCA(n_components=100)
# pca_train.fit(data)
# data_train_pca = pca_train.fit_transform(data)
# print(data_train_pca.shape)
# print(pca_train.explained_variance_ratio_.sum())

# # LCA
# lda_train = LinearDiscriminantAnalysis(n_components=4)
# lda_train.fit(data, labels)
# data_train_lda = lda_train.transform(data)
# print(data_train_lda.shape)
# print(lda_train.explained_variance_ratio_.sum())

# Divide training set and test set 8:2
# x is data, y is labels
split_idx = int(0.8 * len(data))
x_train = data[:split_idx]
x_test = data[split_idx:]

assert len(data) == len(x_train) + len(x_test)

split_idy = int(0.8 * len(labels))
y_train = labels[:split_idy]
y_test = labels[split_idy:]

assert len(data) == len(y_train) + len(y_test)

# Training
# x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=0, train_size=0.8)
# print('训练集和测试集 shape', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = DecisionTreeClassifier(criterion='entropy', random_state=42,
                               class_weight='balanced')  # 实例化模型DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model)  # Output model

# Testing
expected = y_test  # The expected output of the test sample
predicted = model.predict(x_test)  # Test sample prediction

# # 树可视化
# plt.figure(figsize=(40, 20))
# plot_tree(model, filled=True)
# plt.show()

# 输出结果
print(metrics.classification_report(expected, predicted))  # Output results, accuracy, recall rate, f-1 score
print(metrics.confusion_matrix(expected, predicted))  # Confusion matrix

# auc = metrics.roc_auc_score(y_test, predicted)
accuracy = metrics.accuracy_score(y_test, predicted)  # Get accuracy
print("Accuracy: %.2f%%" % (accuracy * 100.0))
