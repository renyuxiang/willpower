
"""
tensor的操作
pytorch的张量数学运算看2_math_cal.png
另外还有torch.addcdiv()、torch.addcmul()
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

def tensor_operation():
    t = torch.randint(3, 10, (2,3,4))
    print(t.size(), '\n')
    print('1.拼接相关')
    print('1.1.torch.cat()')
    # dim=n,表示在n维扩展
    print('dim=0', torch.cat([t, t], dim=0).size())
    print('dim=1', torch.cat([t, t], dim=1).size())
    print('dim=2', torch.cat([t, t], dim=2).size(), '\n')

    print('1.2.torch.stack()')
    # stack会创建新的维度，在新的维度上进行拼接
    t_1_2 = torch.randint(3, 10, (2,3))
    print(t_1_2)
    print('dim=3', torch.stack([t_1_2, t_1_2], dim=2).size())
    print(torch.stack([t_1_2, t_1_2], dim=2))

    print('2.切分相关')
    print('2.1.torch.chunk()')
    t_2_1 = torch.ones((2,5))
    list_of_tensor_2_1 = torch.chunk(t_2_1, dim=0, chunks=2)
    print(t_2_1)
    print([temp for temp in list_of_tensor_2_1])
    print('\n2.2.torch.split()')
    # split功能更强大，可以指定切分的长度
    t_2_2 = torch.ones((2, 5))
    print(t_2_2)
    list_of_tensor_2_2 = torch.split(t_2_2, 2, dim=1)
    print([temp.size() for temp in list_of_tensor_2_2])
    print('\n2.2.1.torch.split()按照指定长度切分')
    list_of_tensor_2_2_1 = torch.split(t_2_2, [2,2,1], dim=1)
    print([temp.size() for temp in list_of_tensor_2_2_1])

    print('\n3.张量索引：')
    print('3.1.torch.index_select:')
    t_3_1 = torch.randint(0, 9, size=(3,3))
    idx_3_1 = torch.tensor([0, 2], dtype=torch.long)    # 索引必须为long
    t_select_3_1 = torch.index_select(t_3_1, dim=0, index=idx_3_1)
    print(t_3_1)
    print(idx_3_1)
    print(t_select_3_1)

    print('\n3.2.torch.masked_select(input, mask):') # 按mask里为True的进行索引,返回1维张量
    t_3_2 = torch.randint(0, 9, (3,3))
    mask_3_2 = t_3_2.ge(5) # ge(5) >=5， gt(5) > 5， le(5) <= 5, lt(5) < 5
    t_select_3_2 = torch.masked_select(t_3_2, mask=mask_3_2)
    print(t_3_2)
    print(mask_3_2)
    print(t_select_3_2)


    print('\n4.张量变换：')
    print('4.1.torch.reshape()')
    # 新张量在内存中是连续的，新张量与input共享数据内存
    t_4_1 = torch.randperm(8)
    t_reshape_4_1 = torch.reshape(t_4_1, (-1, 2, 2))
    print(t_4_1)
    print(t_reshape_4_1)
    t_4_1[0] = 1024
    print(t_reshape_4_1)    # 内存共享

    print('\n4.2.torch.transpose(input=,dim1=,dim2=)')  # 交换张量的2个维度
    t_4_2 = torch.rand(2, 3, 4) # c * h *w -> c * w * h
    print(t_4_2.size())
    transpose_4_2 = torch.transpose(t_4_2, dim0=1, dim1=2)
    print(transpose_4_2.size())
    # torch.t是二维张量

    print('\n4.3.torch.squeeze()')  # 压缩长度为1的维度
    # 若dim=None，会移除所有长度为1的轴
    t_4_3 = torch.rand((1,2,3,1))
    t_sq_4_3 = torch.squeeze(t_4_3)
    t_dim0_4_3 = torch.squeeze(t_4_3, dim=0)
    t_dim1_4_3 = torch.squeeze(t_4_3, dim=1)
    print(t_4_3.size())
    print(t_sq_4_3.size())
    print(t_dim0_4_3.size())
    print(t_dim1_4_3.size())

    print('\n4.4.torch.unsqueeze()')    # 扩展
    t_4_4 = torch.rand((2, 2))
    t_dim0_4_4 = torch.unsqueeze(t_4_4, dim=0)
    t_dim1_4_4 = torch.unsqueeze(t_4_4, dim=1)
    t_dim2_4_4 = torch.unsqueeze(t_4_4, dim=2)
    print(t_4_4.size())
    print(t_dim0_4_4.size())
    print(t_dim1_4_4.size())
    print(t_dim2_4_4.size())

def math_operation():
    # 数学运算
    print('\n5.1.torch.add()')
    t_5_1_temp_0 = torch.randint(3,6, (2,2))
    t_5_1_temp_1 = torch.randint_like(t_5_1_temp_0, 3, 6)
    # t_5_1_temp_0 + 10 * t_5_1_temp_1
    t_add_5_1 = torch.add(t_5_1_temp_0, 10, t_5_1_temp_1)
    print(t_5_1_temp_0)
    print(t_5_1_temp_1)
    print(t_add_5_1)

def linear_regression():
    """
    线性回归
    1.确定模型：y = wx + b ,求解w,b
    2.选择损失函数:mse = 1/m [(y-y_hat)^2]
    3.求解w，b ，w - learning_rate *  w.grad, b - learning_rate *  b.grad

    体现出优化器的重要性
    :return:
    """
    print('线性回归：')
    x = torch.rand(20, 1)
    y_noise = torch.rand(20, 1)
    y = 2 * x + 5 + y_noise
    print(x)
    print(y)
    lr = 0.1
    w = torch.randn((1), requires_grad=True)
    b = torch.zeros((1), requires_grad=True)

    for index in range(3000):
        wx = torch.mul(x, w)
        y_pred = torch.add(wx, b)
        loss = (0.5 * (y - y_pred) **2).mean()
        loss.backward()
        # 更新参数

        w.data.sub_(lr * w.grad)
        b.data.sub_(lr * b.grad)

        if index % 40 == 0:
            plt.scatter(x.data.numpy(), y.data.numpy(), c='blue')
            plt.scatter(x.data.numpy(), y_pred.data.numpy(), c='red')
            plt.plot(x.data.numpy(), y_pred.data.numpy(), c='green', lw=5)
            plt.text(0, 1, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.xlim(0, 1)
            plt.ylim(0, 16)
            plt.title("Iteration:{}\nw:{} b:{}".format(index, w.data.numpy(), b.data.numpy()))
            # plt.show()
            plt.pause(0.5)

        if loss.data.numpy() <= 0.01:
            break
    print(w)
    print(b)


if __name__ == '__main__':
    # tensor_operation()
    # math_operation()
    linear_regression()