#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
from util.common import get_train_test_data, w2excle, Data2Vector

r_dir = 'E:\me\\数据\\蚂蚁金服相似度问句\\train\\'
def split():
    data = get_train_test_data(r_dir+'all.xlsx', cols=3)
    head = data[0]
    data = data[1:]
    print(head)
    print(len(data))
    label_0 = [temp for temp in data if int(temp[2]) == 0]
    label_1 = [temp for temp in data if int(temp[2]) == 1]
    train_set = []
    val_set = []
    test_set = []

    train_set.extend(label_0[: int(len(label_0) * 0.8)])
    train_set.extend(label_1[: int(len(label_1) * 0.8)])

    val_set.extend(label_0[int(len(label_0) * 0.8) : int(len(label_0) * 0.9)])
    val_set.extend(label_1[int(len(label_1) * 0.8) : int(len(label_1) * 0.9)])

    test_set.extend(label_0[int(len(label_0) * 0.9) : ])
    test_set.extend(label_1[int(len(label_1) * 0.9) : ])

    random.shuffle(train_set)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    random.shuffle(test_set)
    import openpyxl
    wb = openpyxl.Workbook()
    file_name = r_dir + 'split_data.xlsx'
    w2excle(train_set, 'train', 3, head=head, wb=wb)
    w2excle(val_set, 'val', 3, head=head, wb=wb)
    w2excle(test_set, 'test', 3, file_name=file_name, head=head, wb=wb)



def stat_data():
    # 统计数据
    char_dic = Data2Vector.generate()
    data = get_train_test_data(r_dir + 'all.xlsx', cols=3)
    not_exist = set()
    max_len = 0
    max_sen = ''
    max_100_q = 0
    max_100_a = 0
    for index, temp in enumerate(data):
        if index % 1000 == 0:
            print(index)
        q = temp[0]
        a = temp[1]
        if len(a) > max_len:
            max_len = len(a)
            max_sen = a
        if len(q) > max_len:
            max_len = len(q)
            max_sen = q
        if len(q) > 80:
            max_100_q += 1
        if len(a) > 80:
            max_100_a += 1
        for _v in q:
            if not char_dic.word2idx.get(_v):
                not_exist.add(_v)
    print(len(not_exist))
    print(not_exist)
    print('ok')
    print('max_len', max_len)
    print('max_sen', max_sen)
    print('max_100_q:', max_100_q)
    print('max_100_a:', max_100_a)

epoch = 30
def run():
    import torch
    a = torch.Tensor([3,4,5,1,2])
    result = torch.argsort(a)
    print(torch.sort(a))
    print(result)

if __name__ == '__main__':
    # split()
    # stat_data()
    run()