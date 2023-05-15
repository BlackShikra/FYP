import csv

import matplotlib.pyplot as plt
import pywt
import numpy as np

# # 取出分类生成字典
# AAMI_key = {'N', 'S', 'V', 'F', 'Q'}
# AAMI = dict()
#
#
# def Tolist(x):
#     return list(map(float, x))
#
#
# for key in AAMI_key:
#     with open(f'{key}.csv', 'r') as f:
#         reader = csv.reader(f)
#         AAMI[key] = list(map(Tolist, list(reader)))

ecg = pywt.data.ecg() # 生成心电信号

w = pywt.Wavelet('db4') # 选用Daubechies8小波

#获取小波分解的最大层级
maxlev = pywt.dwt_max_level(len(ecg), w.dec_len)
coeffs = pywt.wavedec(ecg,'db4', level=maxlev) # 将信号进行小波分解
datarec = pywt.waverec(coeffs, 'db4') # 将信号进行小波重构

# datarec1 = pywt.waverec(np.multiply(coeffs,[1, 0, 0, 0, 0, 0, 0]).tolist(), 'db8') # 将信号进行小波重构
# datarec2 = pywt.waverec(np.multiply(coeffs,[0, 1, 0, 0, 0, 0, 0]).tolist(), 'db8') # 将信号进行小波重构
# datarec3 = pywt.waverec(np.multiply(coeffs,[0, 0, 1, 0, 0, 0, 0]).tolist(), 'db8') # 将信号进行小波重构
# datarec4 = pywt.waverec(np.multiply(coeffs,[0, 0, 0, 1, 0, 0, 0]).tolist(), 'db8') # 将信号进行小波重构
# datarec5 = pywt.waverec(np.multiply(coeffs,[0, 0, 0, 0, 1, 0, 0]).tolist(), 'db8') # 将信号进行小波重构
# datarec6 = pywt.waverec(np.multiply(coeffs,[0, 0, 0, 0, 0, 1, 0]).tolist(), 'db8') # 将信号进行小波重构
# datarec7 = pywt.waverec(np.multiply(coeffs,[0, 0, 0, 0, 0, 0, 1]).tolist(), 'db8') # 将信号进行小波重构
# datarec_re = pywt.waverec(np.multiply(coeffs,[0, 1, 1, 1, 1, 1, 0]).tolist(), 'db8') # 将信号进行小波重构

datarec1 = pywt.waverec(np.multiply(coeffs,[1, 0, 0, 0, 0]).tolist(), 'db4') # 将信号进行小波重构
datarec2 = pywt.waverec(np.multiply(coeffs,[0, 1, 0, 0, 0]).tolist(), 'db4') # 将信号进行小波重构
datarec3 = pywt.waverec(np.multiply(coeffs,[0, 0, 1, 0, 0]).tolist(), 'db4') # 将信号进行小波重构
datarec4 = pywt.waverec(np.multiply(coeffs,[0, 0, 0, 1, 0]).tolist(), 'db4') # 将信号进行小波重构
datarec5 = pywt.waverec(np.multiply(coeffs,[0, 0, 0, 0, 1]).tolist(), 'db4') # 将信号进行小波重构
datarec_re = pywt.waverec(np.multiply(coeffs,[0, 1, 1, 1, 0]).tolist(), 'db4') # 将信号进行小波重构

plt.subplot(9, 1, 1)
plt.plot(datarec1)
plt.subplot(9, 1, 2)
plt.plot(datarec2)
plt.subplot(9, 1, 3)
plt.plot(datarec3)
plt.subplot(9, 1, 4)
plt.plot(datarec4)
plt.subplot(9, 1, 5)
plt.plot(datarec5)
# plt.subplot(9, 1, 6)
# plt.plot(datarec6)
# plt.subplot(9, 1, 7)
# plt.plot(datarec7)
plt.subplot(9, 1, 8)
plt.plot(datarec_re)
plt.subplot(9, 1, 9)
plt.plot(datarec)

plt.show()