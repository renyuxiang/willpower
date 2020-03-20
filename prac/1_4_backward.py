


"""
torch.autograd.backward(tensors, grad_tensors=None,retain_graph=None,create_graph=False)自动求取梯度
tensors:用户求导的张量，如loss
retain_graph：保存计算图
create_graph：创建导数计算图，用于高阶求导
grad_tensors:多梯度权重

tips:
1.梯度不会自动清零
2.依赖于叶子结点的结点，requires_grad默认为True3
3.叶子节点不可执行in-place，
    为什么叶子结点不可以执行in-place操作：
        反向传播需要用到叶子结点的值，前向传播会记录叶子结点的地址，反向传播是根据地址来获取对应的数据，
        若反向传播之前改变了地址当中的数据，那么梯度求解肯定会出错。

"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

def check_grad_tensors():
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)

    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    y.backward()
    print(w.grad)
    print(x.grad)


def check_create_graph():
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2)

    grad_1 = torch.autograd.grad(y, x, create_graph=True)
    print(grad_1)

    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)

def logistic_regression():
    """
    线性2分类模型
    线性回归是分析自变量x与因变量y(标量)之间关系的方法
    逻辑回归是分析自变量x与因变量y(概率)之间关系的方法
    """
    sample_nums = 100
    mean_value = 1.7
    bias = 1
    n_data = torch.ones(sample_nums, 2)
    x0 = torch.normal(mean_value * n_data, 1) + bias
    y0 = torch.zeros(sample_nums)

    x1 = torch.normal(-mean_value * n_data, 1) + bias
    y1 = torch.ones(sample_nums)

    train_x = torch.cat((x0, x1), 0)
    train_y = torch.cat((y0, y1), 0)

    print(train_x.size())
    print(train_y.size())

    class LR(nn.Module):
        def __init__(self):
            super(LR, self).__init__()
            self.features = nn.Linear(2, 1)

        def forward(self, x):
            output = self.features(x)
            return torch.sigmoid(output)

    lr_net = LR()
    lr = 0.01
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)
    for iteration in range(1000):
        y_pred = lr_net(train_x)
        optimizer.zero_grad()

        loss = loss_fn(y_pred, train_y)

        loss.backward()

        optimizer.step()

        if iteration % 50 == 0:
            mask = y_pred.ge(0.5).float().squeeze()
            correct = (mask == train_y).sum()
            acc = correct.item() / train_y.size(0)
            plt.scatter(x0.data.numpy()[:,0], x0.data.numpy()[:,1], c='red', label='class:0')
            plt.scatter(x1.data.numpy()[:,0], x1.data.numpy()[:,1], c='blue', label='class:1')

            w0, w1 = lr_net.features.weight[0]
            w0, w1 = float(w0.item()), float(w1.item())

            plot_b = float(lr_net.features.bias[0].item())
            plot_x = np.arange(-6,6,0.1)
            plot_y = (-w0 * plot_x - plot_b) / w1

            plt.xlim(-5, 7)
            plt.ylim(-7, 7)
            plt.plot(plot_x, plot_y)

            plt.text(-5,5,'loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'green'})

            plt.title('Iteration: {}\nw0:{:.2f} w1:{:.2f} b:{:.2f} accuracy:{:.2%}'.format(
                iteration, w0, w1, plot_b, acc))
            plt.legend()

            plt.show()
            plt.pause(0.5)

            # if acc > 0.99:
            #     break


if __name__ == '__main__':
    # check_create_graph()
    logistic_regression()