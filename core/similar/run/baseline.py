#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from util.common import get_train_test_data, w2excle, Data2Vector
from core.similar.model.siamese_lstm import SiameseLstm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
random.seed(100)

r_dir = 'E:\me\\数据\\蚂蚁金服相似度问句\\train\\'
epochs = 5
embed_size = 128
hidden_size = 128
lr = 0.01
min_loss = None
every_iterator_print = 2
threshold = 0.5


char_dic = Data2Vector.generate()


class QuestionsDataset(Dataset):
    def __init__(self, sheet_index, dic):
        self.data = get_train_test_data(r_dir+'s_split_data.xlsx', sheet_index, start_line=1, cols=3)
        self.dic = dic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        questions_pair = self.data[index]
        q1 = self.dic.get_vector(questions_pair[0])
        q2 = self.dic.get_vector(questions_pair[1])
        return q1, q2, int(questions_pair[2])

def sort_by_lengths(datas, length_list):
    indices = sorted(range(len(length_list)),
                     key=lambda k: length_list[k],
                     reverse=True)
    result = []
    for index, data in enumerate(datas):
        result.append([data[i] for i in indices])
    return result, indices

def custom_collate(batch):
    # batch = list of tuples where each tuple is of the form ([i1, i2, i3], [j1, j2, j3], label)
    q1_list = []
    q2_list = []
    labels = []
    for training_example in batch:
        q1_list.append(training_example[0])
        q2_list.append(training_example[1])
        labels.append(training_example[2])
    q1_len = [len(q) for q in q1_list]
    q2_len = [len(q) for q in q2_list]
    q1_max_len = max(q1_len)
    q2_max_len = max(q2_len)
    [temp.extend([0] * (q1_max_len - len(temp)) ) for temp in q1_list]
    [temp.extend([0] * (q2_max_len - len(temp))) for temp in q2_list]
    return q1_list, q1_len, q2_list, q2_len, labels


class Trainer(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SiameseLstm(embed_size, hidden_size, self.device).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.threshold = torch.Tensor([threshold]).to(self.device)
        self.min_loss = None
        self.model_best_name = 'siamese_best.pkl'
        self.model_name = 'siamese.pkl'

    def train(self, train_loader, val_loader):
        for epoch in range(1, epochs + 1):
            train_result = self.train_step(epoch, train_loader)
            val_result= self.val(epoch, val_loader)
            print('epoch: {}, train acc : {} / {} = {} %, train loss: {:.8f}; val acc : {} / {} = {} %, val loss: {:.8f}'.format(
                epoch,train_result['correct'], train_result['total'], train_result['acc'], train_result['loss'],
                val_result['correct'],val_result['total'],val_result['acc'],val_result['loss']))
            val_loss = val_result['loss']
            if self.min_loss is None or (self.min_loss and self.min_loss > val_loss):
                torch.save(self.model, self.model_best_name)
                print('epoch:%s 保存新参数, val loss from %s to %s' % (epoch, self.min_loss, val_loss))
                self.min_loss = val_loss
            print()
        torch.save(self.model, self.model_name)
        print('train ok')

    def train_step(self, epoch, train_loader):
        device = self.device
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        threshold = self.threshold

        model.train()
        total_step = len(train_loader)
        total = 0
        correct = 0
        losses = []
        result = {}

        with tqdm(total=total_step) as pbar:
            iterator_loss = []
            for i, (q1, q1_len, q2, q2_len, label) in enumerate(train_loader):
                q1 = torch.LongTensor(q1).to(device)
                q1_len = torch.LongTensor(q1_len).to(device)
                q2 = torch.LongTensor(q2).to(device)
                q2_len = torch.LongTensor(q2_len).to(device)
                label = torch.Tensor(label).to(device)

                output = model(q1, q1_len, q2, q2_len)
                optimizer.zero_grad()
                loss = criterion(output, label)
                iterator_loss.append(loss.item())
                losses.append(loss.item())

                predictions = (output > threshold).float() * 1
                correct += (predictions == label).sum().item()
                total += label.size(0)

                loss.backward()
                optimizer.step()
                if (i + 1) % every_iterator_print == 0:
                    pbar.set_postfix({'Epoch': '[{}/{}]'.format(epoch, epochs),
                                      'Step': '[{}/{}]'.format(i + 1, total_step),
                                      'mean loss': '{:.8f}'.format(np.mean(iterator_loss)),
                                      })
                    pbar.update(every_iterator_print)
                    iterator_loss = []
        acc = 100 * correct / total
        loss_mean = np.mean(losses)
        result.update({'acc': acc, 'loss': loss_mean, 'correct': correct, 'total': total})
        return result


    def val(self, epoch, loader):
        device = self.device
        model = self.model
        criterion = self.criterion
        threshold = self.threshold
        model.eval()
        total = 0
        correct = 0
        losses = []
        result = {}
        for i, (q1, q1_len, q2, q2_len, label) in enumerate(loader):
            q1 = torch.LongTensor(q1).to(device)
            q1_len = torch.LongTensor(q1_len).to(device)
            q2 = torch.LongTensor(q2).to(device)
            q2_len = torch.LongTensor(q2_len).to(device)
            label = torch.Tensor(label).to(device)

            output = model(q1, q1_len, q2, q2_len)

            loss = criterion(output, label)
            losses.append(loss.item())

            predictions = (output > threshold).float() * 1
            correct += (predictions == label).sum().item()
            total += label.size(0)

        acc = 100 * correct / total
        loss_mean = np.mean(losses)
        result.update({'acc': acc, 'loss': loss_mean, 'correct': correct, 'total': total})
        return result


    def t(self, loader, model=None, name=None):
        check_model = model
        if name:
            check_model = torch.load(name)
        check_model.eval()
        device = self.device
        threshold = self.threshold
        total_step = len(loader)
        total = 0
        correct = 0
        result = []
        with tqdm(total=total_step) as pbar:
            for i, (q1, q1_len, q2, q2_len, label) in enumerate(loader):
                q1 = torch.LongTensor(q1).to(device)
                q1_len = torch.LongTensor(q1_len).to(device)
                q2 = torch.LongTensor(q2).to(device)
                q2_len = torch.LongTensor(q2_len).to(device)
                label = torch.Tensor(label).to(device)

                output = model(q1, q1_len, q2, q2_len)

                predictions = (output > threshold).float() * 1
                correct += (predictions == label).sum().item()
                total += label.size(0)
                if (i + 1) % every_iterator_print == 0:
                    pbar.update(every_iterator_print)
        acc = 100 * correct / total
        print('test acc:%s' % acc)

# def t(model=None, name=None):
#     check_model = model
#     if name:
#         check_model = torch.load(name)
#     check_model.eval()
#     threshold = 0.5
#     total = 0
#     correct = 0
#     result = [] # 0-not equal, 1-equal
#     for (x, y, raw_x) in test_data:
#         prop, predicted, is_change = check_model.predict(x, threshold=threshold)
#         result.append((raw_x, y, predicted, is_equal, prop, is_change))
#     print('test finish!!!')


if __name__ == '__main__':
    trainer = Trainer()
    train_dataset = QuestionsDataset(sheet_index=0, dic=char_dic)
    val_dataset = QuestionsDataset(sheet_index=1, dic=char_dic)
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    trainer.train(train_loader, val_loader)