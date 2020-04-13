"""
回归
"""

import keras.backend as K
import numpy as np
np.random.seed(1337)  # for reproducibility
# from keras.models import Sequential
# from keras.layers import Dense
# import matplotlib.pyplot as plt # 可视化模块
#
# # 创建训练数据
# X = np.linspace(-1, 1, 200)
# print(X)

a = np.array([[1,2, 3],[3,4,5]])
sum0 = np.sum(a, axis=0, keepdims=True)
sum1 = np.sum(a, axis=1, keepdims=True)
print(a)
print(a.shape)
print(sum0)
print(sum1)
print(K.backend())
print(K.image_data_format())