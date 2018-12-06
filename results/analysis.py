import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import numpy as np

lstm = open("lstm_errors.txt")
cnn_lstm = open("cnn_lstm_errors.txt")
lstm = lstm.read().split(",")[:-1]
cnn_lstm = cnn_lstm.read().split(",")[:-1]
lstm = [float(e) for e in lstm][:2000]
cnn_lstm = [float(e) for e in cnn_lstm]
cnn_lstm = [cnn_lstm[i] for i in range(len(cnn_lstm)) if i % 10 == 0][:2000]
smoothed_lstm = gaussian_filter(lstm, sigma=30)
smoothed_cnn_lstm = gaussian_filter(cnn_lstm, sigma=30)
plt.plot(smoothed_lstm, label="lstm", color="red")
plt.plot(smoothed_cnn_lstm, label="cnn_lstm", color="blue")
plt.scatter([i for i in range(len(lstm))], lstm, color="red", alpha=0.02)
plt.scatter([i for i in range(len(cnn_lstm))], cnn_lstm, color="blue", alpha=0.02)
plt.xlabel("time")
plt.ylabel("error (MSE)")
plt.ylim(0, 150)
plt.legend()
plt.show()