import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
# Shuffle
data_set = data_set.sample(frac=1.0)
print(data_set)

# Read data and labels
data = pd.DataFrame(data_set.iloc[:, :260])
labels = pd.DataFrame(data_set.iloc[:, 260])
print(data)
print(labels)

# PCA
pca_train = PCA(n_components=29)
pca_train.fit(data)
data_train_pca = pca_train.transform(data)
print(data_train_pca.shape)
print(pca_train.explained_variance_ratio_.sum())

# LCA
lda_train = LinearDiscriminantAnalysis(n_components=4)
lda_train.fit(data, labels)
data_train_lca = lda_train.transform(data)
print(data_train_lca.shape)
print(lda_train.explained_variance_ratio_.sum())
