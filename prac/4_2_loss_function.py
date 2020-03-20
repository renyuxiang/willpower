"""

损失函数：衡量模型输出与真实标签的差异

损失函数(loss function): 聚焦于单个样本
代价函数(cost function): 聚焦于全部样本

目标函数(Objective function): obj = cost + regularization
不光要拟合数据，对模型也要进行约束


交叉熵 = 信息熵 + 相对熵(因为H(p)是常量，所以又称KL散度)

weight
"""

import torch
import torch.nn as nn

nn.CrossEntropyLoss()

if __name__ == '__main__':
    pass