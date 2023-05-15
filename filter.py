import csv

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy import signal

# Get Dictionary
AAMI_key = {'N', 'S', 'V', 'F', 'Q'}
AAMI = dict()


def Tolist(x):
    return list(map(float, x))

for key in AAMI_key:
    with open(f'{key}.csv', 'r') as f:
        reader = csv.reader(f)
        AAMI[key] = list(map(Tolist, list(reader)))


# for i in AAMI_key:
#     print(f'{i}={np.array(AAMI[i]).shape}')
# Frequence is the sampling frequency of the signal
# highpass The highest frequency through which a low-pass filter can pass
# lowpass  The lowest frequency through which a high-pass filter can pass

# Bandpass
def bandpass_filter(data, frequency=360, highpass=40, lowpass=0.5):
    [b, a] = signal.butter(3, [lowpass / frequency * 2, highpass / frequency * 2], 'bandpass')
    modi_signal = signal.filtfilt(b, a, data)
    return modi_signal


# DWT
def wavelet_decomposition_filter(data):
    w = pywt.Wavelet('db8')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)
    datarec = pywt.waverec(coeffs, 'db8')

    datarec1 = pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0, 0, 0]).tolist(), 'db8')
    datarec2 = pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0, 0, 0]).tolist(), 'db8')
    datarec3 = pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0, 0, 0]).tolist(), 'db8')
    datarec4 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0, 0, 0]).tolist(), 'db8')
    datarec5 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1, 0, 0]).tolist(), 'db8')
    datarec6 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 0, 1, 0]).tolist(), 'db8')
    datarec7 = pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 0, 0, 1]).tolist(), 'db8')

    # The highest and lowest frequencies were removed and reconstructed
    modi_signal = pywt.waverec(np.multiply(coeffs, [0, 1, 1, 1, 1, 1, 0]).tolist(), 'db8')
    return modi_signal


# Filter visualization
# n = 1
# for list in AAMI['S']:
#     ecg = list  # 成心电信号
#     filter_ecg = bandpass_filter(ecg)  # Bandpass
#     # filter_ecg = wavelet_decomposition_filter(ecg)  # DWT
#     # print
#     plt.plot(ecg)
#     plt.plot(filter_ecg)
#     plt.legend(['Before', 'After'])
#     plt.show()
#     print('画完了%d' % n + '张')
#     n = n + 1

# Save to csv
for key, value in AAMI.items():
    with open(f'modiData/bandpass_filter/{key}.csv', 'w',newline='\n') as f:
        writer = csv.writer(f)
        # Write each piece of data in the list to a csv file, separated by commas
        # The data passed in is a nested list or tuple within the list, each list or tuple is the data for each row
        bandpass_signal = bandpass_filter(value)
        writer.writerows(bandpass_signal)

    # with open(f'modiData/DWT_filter/{key}.csv', 'w',newline='\n') as f:
    #     writer = csv.writer(f)
    #     # Write each piece of data in the list to a csv file, separated by commas
    #     # The data passed in is a nested list or tuple within the list, each list or tuple is the data for each row
    #     w_decomp_signal = bandpass_filter(value)
    #     writer.writerows(w_decomp_signal)
