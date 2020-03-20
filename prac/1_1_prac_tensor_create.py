
"""
reference：https://www.bilibili.com/video/av81064297?p=3
https://msd.misuland.com/pd/3181438578597040026
一、历程

1.2017.1正式发布pytorch
2.2018.4更新到0.4.0版，支持windows，caffe2正式并入pytorch
3.2018.11更新到1.0稳定版，为github增长第二块的开源项目
4.2019.5更新到1.1.0，支持Tensorboard，增强可视化功能
5.2019.8更新到1.2.0版本，更新torchvision、torchaudio、torchtext

二、Pytorch相关概念
Variable主要用于封装Tensor，进行自动求导
Variabl在0.4.0以后被并入Tensor里，后面不需要关心Variable。

Variable属性(0.4.0以前的版本)：
    data：被包装的Tensor
    grad：data的梯度
    grad_fn：创建Tensor的Function，是自动求导的关键
    requires_grad：指示是否需要梯度
    is_leaf：指示是否是叶子结点（计算图里会用到）

Variable新增属性(0.4.0以后的版本)：
    dtype:张量的数据类型(9种)，如torch.FloatTensor、torch.cuda.FloatTensor
    shape:形状
    device：张量所在的设备，GPU/CPU,是加速的关键

三、Tensor概念

1.创建：
    1.直接创建
    2.依据数值创建
    3.依据概率创建
"""
import torch
import numpy as np

def create_tensor():
    # 创建tensor
    print('1.1.直接创建Tensor:')
    tensor_1 = torch.ones((3, 3))
    print(tensor_1, '\n')

    print('1.2.从numpy创建Tensor：')
    numpy_2 = np.array([[1,2,3], [4,5,6]])
    print('numpy:', numpy_2)
    tensor_2 = torch.from_numpy(numpy_2)
    print('tensor:\n', tensor_2, '\n')


    print('2.1.依据数值创建：zeros()：')
    print(torch.zeros(3,3), '\n')
    print('2.2.zeros_like()')
    print(torch.zeros_like(tensor_2), '\n')
    print('2.3.填充的方式torch.full():')
    print(torch.full([2,3], 10), '\n')

    print('2.4.等差数列的方式torch.arange(),[),end是取不到的:')
    print(torch.arange(start=3, end=9, step=3), '\n')

    print('2.5.创建均分的1维张量torch.linspace(),end取得到')
    # steps表示分成几个，步长计算公式: ( end - start ) / steps
    print(torch.linspace(start=1, end=2, steps=5), '\n')

    print('2.6对数均分数列torch.logspace(), 依赖base,默认为10')
    print(torch.logspace(start=1, end=2, steps=5), '\n')
    print('第4个值:', np.power(10, 1.75))

    print('2.7单位对角矩阵torch.eye():')
    print(torch.eye(n=5), '\n')


    print('3.依据概率分布创建张量')
    print('3.1.依据正态/高斯分布创建torch.normal(mean=,std=):')
    """
    mean和std可以为张量或标量，有4种模式
    """
    print('3.1.1.mean:张量, std:张量：')
    mean_3_1 = torch.arange(1, 5, dtype=torch.float)
    std_3_1 = torch.arange(1, 5, dtype=torch.float)
    print(mean_3_1)
    # 第一个数是在mean=1和std=1的正态分布里采样得到，第4个数是在mean=4和std=4里采样得到
    print(torch.normal(mean_3_1, std_3_1), '\n')

    print('3.2,随机生成正态分布torch.randn()：')
    print(torch.randn((3,3)), '\n')

    print('3.3,随机生成均匀分布torch.randint()：')
    print(torch.randint(3, 10, (3, 3)), '\n')
    print(torch.randint(5, (3,), dtype=torch.int64), '\n')

    print('3.4,生成0-(n-1)的随机排列，torch.randperm()，常用作索引：')
    print(torch.randperm(10), '\n')

    print('3.5,生成伯努利分布，torch.bernoulli()，常用作索引：')
    # 0-1分布
    generator_3_5 = torch.empty(3, 3).uniform_(0, 1)    # 生成分布
    print(generator_3_5)
    print(torch.bernoulli(generator_3_5), '\n')
    print('ok')

if __name__ == '__main__':
    create_tensor()