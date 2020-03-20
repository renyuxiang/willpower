import numpy as np

def Batchnorm(x, gamma, beta, bn_param=None):

    # x_shape:[B, C, H, W]
    # running_mean = bn_param['running_mean']
    # running_var = bn_param['running_var']
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(0, 2, 3), keepdims=True)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta

    # 因为在测试时是单个图片测试，这里保留训练时的均值和方差，用在后面测试时用
    # running_mean = momentum * running_mean + (1 - momentum) * x_mean
    # running_var = momentum * running_var + (1 - momentum) * x_var

    # bn_param['running_mean'] = running_mean
    # bn_param['running_var'] = running_var

    return results

def check_BN():
    x = np.random.randint(0, 9, (3, 3, 2, 2))
    print(x)
    Batchnorm(x, 0.9, 0.1)


if __name__ == '__main__':
    check_BN()