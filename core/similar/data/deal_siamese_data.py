#!/usr/bin/env python
# -*- coding: utf-8 -*-
from willpower.util.common import get_train_test_data, w2excle, Data2Vector
from torch.utils.data import Dataset, DataLoader

r_dir = 'E:\me\\数据\\蚂蚁金服相似度问句\\train\\'
r_dir = '/home/renyx/data/similar/'

class QuestionsDataset(Dataset):
    def __init__(self, sheet_index, dic):
        self.data = get_train_test_data(r_dir+'split_data.xlsx', sheet_index, start_line=1, cols=3)
        self.dic = dic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        questions_pair = self.data[index]
        q1 = self.dic.get_vector(questions_pair[0])
        q2 = self.dic.get_vector(questions_pair[1])
        # if len(q1) == 0 or len(q2) == 0:
        #     print(index, questions_pair[0], questions_pair[1])
        return q1, q2, int(questions_pair[2])

def create_siamese_data():
    char_dic = Data2Vector.generate()
    dataset = QuestionsDataset(sheet_index=2, dic=char_dic)
    result = []
    for i, temp in enumerate(dataset.data):
        q1, q2, label = dataset.__getitem__(i)
        result.append('%s\t%s\t%s\n' % (label, ','.join([str(temp) for temp in q1]), ','.join([str(temp) for temp in q2])))
    with open(file='test.txt', mode='w', encoding='utf-8') as f:
        f.writelines(result)


if __name__ == '__main__':
    create_siamese_data()