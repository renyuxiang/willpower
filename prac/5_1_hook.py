import torch
from torch.autograd import Variable


def hook_check():
    grad_list = []

    def print_grad(grad):
        print('aa')
        grad_list.append(grad)
    a = torch.randn((2, 1))
    q = a.mean(-1, keepdim=True)

    print(q)
    x = torch.randn((2, 1), requires_grad=True)
    y = x + 2
    z = torch.mean(torch.pow(y, 2))
    lr = 1e-3
    y.register_hook(print_grad)
    z.backward()
    x.data -= lr * x.grad.data
    print(x.data)


if __name__ == '__main__':
    hook_check()