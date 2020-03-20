import torch

import torch.nn as nn

from  torch.optim.sgd import SGD
class ZeroModel(nn.Module):

    def __init__(self):
        super(ZeroModel, self).__init__()

        self.linear_1 = nn.Linear(2, 2)
        self.linear_2 = nn.Linear(2, 2)
        self.linear_3 = nn.Linear(2, 2)
        self.linear_4 = nn.Linear(2, 2)
        self.activation = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.init_weight()
        print('aa')

    def init_weight(self):
        torch.nn.init.constant_(self.linear_1.weight, 0)
        torch.nn.init.constant_(self.linear_2.weight, 0)
        torch.nn.init.constant_(self.linear_3.weight, 0)
        torch.nn.init.constant_(self.linear_4.weight, 0)

        torch.nn.init.constant_(self.linear_1.bias, 1)
        torch.nn.init.constant_(self.linear_2.bias, 1)
        torch.nn.init.constant_(self.linear_3.bias, 1)
        torch.nn.init.constant_(self.linear_4.bias, 1)


    def forward(self, input):
        x1 = self.linear_1(input)
        x1_1 = self.activation(x1)

        x2 = self.linear_1(x1_1)
        x2_1 = self.activation(x2)

        x3 = self.linear_1(x2_1)
        x3_1 = self.activation(x3)

        x4 = self.linear_1(x3_1)
        x4_1 = self.activation(x4)

        return x4_1

if __name__ == '__main__':
    model = ZeroModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()
    in_ = torch.Tensor([2, 3])
    label = torch.Tensor([0.3])

    for temp in range(10):
        optimizer.zero_grad()
        result = model(in_)
        print(result)
        loss = torch.sum(result - label)
        loss.backward()
    print('ok')

