"""
主要看不量的权值初始化是怎么影响梯度爆炸和梯度消失的
看p_4_1_2_grad_explore.png
每一层的隐藏层的方差为n* 1 * 1 = n，n为神经元的个数，第一层就将原来均值为0，标准差为1的扩大到标准差为根号n，方差为n的大小，
若再加几个网络层，会超出数值范围，形成梯度爆炸。

为了让每一层的标准差都为1，让D(w)=1=1/n,需要初始化w，让w的方差为1/n


具有激活函数的权值初始化：

方差一致性，使方差=1
Xavier初始化：针对饱和激活函数(sigmoid、Tanh)，不适用于relu. nn.init.xavier_uniform()
Kaiming初始化：针对Relu. nn.init.kaiming_normal_()

p_4_1_4_init_method：10种权值初始化方法
nn.init.calculate_gain(nonlinearity)，计算激活函数方差变化尺度，输入数据的方差 / 输出数据的方差
"""