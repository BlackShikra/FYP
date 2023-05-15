import pandas as pd
from tsfresh import extract_features

# 数据读取

data = pd.read_csv("modiData/bandpass_filter/F.csv")
# data_F = pd.read_csv("modiData/bandpass_filter/F.csv"
# data_N = pd.read_csv("modiData/bandpass_filter/N.csv")
# data_S = pd.read_csv("modiData/bandpass_filter/S.csv")
# data_V = pd.read_csv("modiData/bandpass_filter/V.csv")
# train_set = [data_F, data_N, data_S, data_V]
#
# time = []
# for n in range(0, 260):
#     time.append(n)
#
# data = pd.DataFrame()
# label = 0
# for df in train_set:
#     # 加入time索引
#     df.loc[-1] = df.columns.tolist()
#     df.index = df.index + 1
#     df.sort_index(inplace=True)
#     df.columns = time
#     df = np.concatenate(([[label]], df), axis=1)
#     data = data.append(df)
#     label = label + 1
#
# print(data.shape)
# print(data.head())
# print(data.tail())

# 增加time属性
time = []
for n in range(0, 260):
    time.append(n)
data.loc[-1] = data.columns.tolist()
data.index = data.index + 1
data.sort_index(inplace=True)
data.columns = time

data['heartbeat_signals'] = data[data.columns[0:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

heartbeat_signal = pd.DataFrame(data.iloc[:, 260])
heartbeat_signal['id'] = range(0, len(heartbeat_signal))
id = heartbeat_signal.pop('id')
heartbeat_signal.insert(0, 'id', id)
heartbeat_signal['label'] = 0
# print(heartbeat_signal)

# 0 - N(正常或者束支传导阻滞节拍)
# 1 - V(心室异常节拍)
# 2 - s(室上性异常节拍)
# 3 - F(融合节拍)
# 对心电特征进行行转列处理，同时为每个心电信号加入时间步特征time
train_heartbeat_df = heartbeat_signal["heartbeat_signals"].str.split(",", expand=True).stack()
# print(train_heartbeat_df)
train_heartbeat_df = train_heartbeat_df.reset_index()
# print(train_heartbeat_df)
train_heartbeat_df = train_heartbeat_df.set_index("level_0")
train_heartbeat_df.index.name = None
train_heartbeat_df.rename(columns={"level_1": "time", 0: "heartbeat_signals"}, inplace=True)
train_heartbeat_df["heartbeat_signals"] = train_heartbeat_df["heartbeat_signals"].astype(float)
# print(train_heartbeat_df)

# 提取标签,重构训练集
data_train = pd.DataFrame(heartbeat_signal)
data_train_label = data_train["label"]
data_train = data_train.drop("label", axis=1)
data_train = data_train.drop("heartbeat_signals", axis=1)
data_train = data_train.join(train_heartbeat_df)
# print(data_train)

# 特征提取
train_features = extract_features(data_train, column_id='id', column_sort='time')
print(train_features)