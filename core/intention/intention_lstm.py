

"""
1.lstm + attention
2.获取attention的值、并可视化
3.attention的方式是 lstm前attention、后attention
"""
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random
import xlrd, openpyxl
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from tqdm import tqdm
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

random.seed(100)
torch.manual_seed(100)
print(torch.version.cuda)

data_dir = '/home/renyx/work_check/model_data/intention/'
# data_dir = 'E:\\me\\data\\intention\\'
lr = 1e-3
# lr = 0.01
epochs = 30
w_path = 'history/3/'
every_iterator_print = 1
all_data_count = None
t_name = 'intention_lstm'

all_intent_dict = {
    '号源是否更新': ('haoyuanRefresh', 0),
    '有没有号': ('register', 1),
    '医生最近号源时间': ('recentHaoyuanTime', 2),
    '选医生-限定词': ('doctor', 3),
    '医生如何': ('doctorQuality', 4),
    '该医院是否有该科室': ('hospitalDepartment', 5),
    '医院如何': ('hospitalQuality', 6),
    '选医院-限定词': ('hospital', 7),
    '选科室下的细分科室': ('departmentSubset', 8),
    '科室二选一': ('departmentAmong', 9),
    '选科室-限定词': ('department', 10),
    '是否这个科室': ('departmentConfirm', 11),
    '附近的医院': ('hospitalNearby', 12),
    '医院排序': ('hospitalRank', 13),
    '其他other': ('other', 14),
    '内容': ('content', 15),
    '客服': ('customerService', 16)
}

# 构建意图字典{0:haoyuanRefresh, 1:register}
num_label_intent_dict = {label_temp: intent_temp for _, (intent_temp, label_temp) in all_intent_dict.items()}
# 构建意图字典{haoyuanRefresh:0, register:1}
label_num_intent_dict = {intent_temp: label_temp for _, (intent_temp, label_temp) in all_intent_dict.items()}



def get_train_test_data(r_path, sheet_index=0, start_line=0, cols=2):
    data = xlrd.open_workbook(r_path)
    table = data.sheets()[sheet_index]
    # 行数
    row_count = table.nrows
    # 列数
    col_count = table.ncols
    print('rows: ', row_count)
    print('cols: ', col_count)
    result = []
    for row_num in range(start_line, row_count):
        row_data = table.row_values(row_num)
        data = []
        col_real_temp = cols
        if len(row_data) < col_real_temp:
            col_real_temp = len(row_data)
        for col_temp in range(0, col_real_temp):
            data_temp = row_data[col_temp]
            data.append(data_temp)
        result.append(data)
    print('result len: %s' % len(result))
    return result

def w2excle(data, sheet_name, col=None, file_name=None, head=None, wb=None):
    if not wb:
        wb = openpyxl.Workbook()
    ws = wb.create_sheet(title=sheet_name)

    num = 0
    if head:
        ws.append(head)
        num += 1
    for index, temp in enumerate(data):
        col_real_temp = col
        if len(temp) < col_real_temp:
            col_real_temp = len(temp)
        for col_temp in range(0, col_real_temp):
            try:
                ws.cell(index + num + 1, col_temp + 1, temp[col_temp])
            except Exception as err:
                print(err)
    if file_name:
        wb.save(file_name)

class IntentionLstm(nn.Module):

    def __init__(self, vocab_size, emb_size, hidden_size, out_size=17):
        super(IntentionLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=2, dropout=0.1)
        self.lin = nn.Linear(2 * hidden_size, out_size)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_ids, lengths, labels=None):
        emb = self.embedding(input_ids)  # [B, L, emb_size]
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _v1 = self.bilstm(packed)
        unpacked, _v2 = pad_packed_sequence(rnn_out, batch_first=True)
        # lstm分类,取最后一个有效输出作为最终输出（0为无效输出）
        last_output = unpacked[torch.LongTensor(range(input_ids.shape[0])), lengths - 1]
        # unpacked_t = F.relu(last_output)
        unpacked_t = last_output
        scores = self.lin(unpacked_t)  # [B, L, out_size]
        # logits = F.softmax(scores, dim=1)
        outputs = (scores,)
        if labels is not None:
            loss = self.criterion(scores, labels)
            outputs = (loss,) + outputs
        return outputs

def custom_collate(batch):
    # batch = list of tuples where each tuple is of the form ([i1, i2, i3], [j1, j2, j3], label)
    q1_list = []
    labels = []
    for training_example in batch:
        q1_temp = training_example[0]
        q1_list.append(q1_temp)
        labels.append(training_example[1])
    indices = sorted(range(len(q1_list)), key=lambda k: len(q1_list[k]), reverse=True)
    q1_list = [q1_list[i] for i in indices]
    labels = [labels[i] for i in indices]
    q1_len = [len(q) for q in q1_list]
    q1_max_len = q1_len[0]
    [temp.extend([0] * (q1_max_len - len(temp)) ) for temp in q1_list]
    # return torch.Tensor(q1_list), torch.Tensor(q1_len) , torch.Tensor(labels)
    return q1_list, q1_len, labels

class Trainer(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_id = None
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            self.device_id = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        print('device:', self.device)
        self.paraller = False
        if self.device_id == 'cuda' and len(self.device_id) >= 2:
            self.paraller = True
        print('paraller:', self.paraller)
        print('start load source_model....')
        self.source_model = IntentionLstm(vocab_size=6000, emb_size=100, hidden_size=128)
        print(self.source_model)
        if self.paraller:
            print('start load model....')
            self.model = nn.DataParallel(self.source_model, device_ids=self.device_id).to(self.device)
            print('load model finish....')
        else:
            self.model = self.source_model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = 0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=4, verbose=True)
        self.min_loss = None
        self.model_best_name = w_path + 'intention_lstm_best.pkl'
        self.model_name = w_path + 'intention_lstm.pkl'

        # no_decay = ['bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
        #      'weight_decay': weight_decay},
        #     {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0}
        # ]
        # optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    def train(self, train_loader, val_loader):
        for epoch in range(1, epochs + 1):
            print()
            start = time.time()
            train_result = self.train_step(epoch, train_loader)
            print('epoch: {}, ms:{}s'.format(epoch, int(time.time()-start)))
            val_result= self.val(epoch, val_loader)
            print('epoch: {}, ms:{}s, train_acc : {} / {} = {} %, train_loss: {:.8f}; val_acc : {} / {} = {} %, val_loss: {:.8f}'.format(
                epoch, int(time.time()-start), train_result['correct'], train_result['total'], train_result['acc'], train_result['loss'],
                val_result['correct'],val_result['total'],val_result['acc'],val_result['loss']))
            val_loss = val_result['loss']
            if self.min_loss is None or (self.min_loss and self.min_loss > val_loss):
                self.save(self.model_best_name)
                print('epoch:%s 保存新参数, val loss from %s to %s' % (epoch, self.min_loss, val_loss))
                self.min_loss = val_loss
            self.scheduler.step(val_loss)

        self.save(self.model_name)
        print('start t...')
        self.t(model=self.model, echo='intention_end')
        print('train ok')

    def save(self, model_name):
        if self.paraller:
            torch.save(self.model.module, model_name)
        else:
            torch.save(self.model, model_name)


    def train_step(self, epoch, train_loader):
        device = self.device
        model = self.model
        optimizer = self.optimizer
        # criterion = self.criterion
        model.train()
        total_step = len(train_loader)
        total = 0
        correct = 0
        losses = 0.0
        loss_step = 1e-5
        result = {}
        with tqdm(total=total_step) as pbar:
            iterator_loss = 0.0
            iterator_step = 1e-5
            for i, batch in enumerate(train_loader):
                batch = tuple(torch.LongTensor(t).to(device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'lengths': batch[1],
                          'labels': batch[2]
                          # 'token_type_ids': batch[2],
                          # 'attention_mask': batch[1]
                          }

                outputs = model(**inputs)
                loss = outputs[0]
                if self.paraller:
                    loss = loss.mean() # mean() to average on multi-gpu parallel training

                iterator_loss += loss.item()
                iterator_step += 1
                losses += loss.item()
                loss_step += 1

                logits = outputs[1]

                preds = torch.argmax(logits.detach().cpu(), dim=1)
                out_label_ids = inputs['labels'].detach().cpu()
                correct += (preds == out_label_ids).sum().item()
                total += out_label_ids.shape[0]

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (i + 1) % every_iterator_print == 0:
                    pbar.set_postfix({'Epoch': '[{}/{}]'.format(epoch, epochs),
                                      'Step': '[{}/{}]'.format(i + 1, total_step),
                                      'loss': '{:.8f}'.format(iterator_loss / iterator_step),
                                      })
                    pbar.update(every_iterator_print)
                    iterator_loss = 0.0
                    iterator_step = 1e-5
        acc = 100 * correct / total
        loss_mean = losses / loss_step
        result.update({'acc': acc, 'loss': loss_mean, 'correct': correct, 'total': total})
        return result


    def val(self, epoch, loader):
        device = self.device
        model = self.model
        model.eval()
        total = 0
        correct = 0
        losses = 0.0
        loss_step = 1e-5
        result = {}
        for i, batch in enumerate(loader):
            batch = tuple(torch.LongTensor(t).to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'lengths': batch[1],
                      'labels': batch[2]
                      }

            outputs = model(**inputs)
            loss = outputs[0]
            if self.paraller:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            losses += loss.item()
            loss_step += 1

            logits = outputs[1]
            preds = torch.argmax(logits.detach().cpu(), dim=1)
            out_label_ids = inputs['labels'].detach().cpu()
            correct += (preds == out_label_ids).sum().item()
            total += out_label_ids.shape[0]

        acc = 100 * correct / total
        loss_mean = losses / loss_step
        result.update({'acc': acc, 'loss': loss_mean, 'correct': correct, 'total': total})
        return result


    def t(self, model=None, name=None, **kwargs):
        check_model = model
        if name:
        #     if self.device == 'cpu':
        #         check_model = torch.load(w_path+name, map_location='cpu')
            check_model = torch.load(w_path+name, map_location=self.device)
        check_model.eval()
        device = self.device
        total = 0
        correct = 0
        result = []
        data_set = IntentionDataset(name=data_dir + 'intention.xls', sheet_index=2, shuffle=False)
        for i in list(range(len(data_set))):
            data_temp = data_set.__getitem__(i)
            inputs = {'input_ids': torch.LongTensor([data_temp[0]]).to(device),
                      'lengths': torch.LongTensor([len(data_temp[0])]).to(device),
                      }
            raw_x = data_set.data[i][0]
            outputs = check_model(**inputs)
            logits = outputs[0]
            pred_score, preds = torch.max(logits.detach().cpu(), dim=1)
            pred_score = pred_score.item()
            preds = preds.item()
            out_label_ids = data_temp[1]

            equal = 0   # 0 - 不相等
            if preds == out_label_ids:
                equal += 1
                correct += 1
            total += 1
            result.append([raw_x, num_label_intent_dict[out_label_ids], out_label_ids, preds, equal, pred_score])
        acc = 100 * correct / total
        print('acc : {} / {} = {} %'.format(correct, total, acc))
        head = ['问句', 'y_cn', 'y', 'y_hat', 'equal', 'score']
        w2excle(result, sheet_name='test_result', col=6, file_name=w_path + kwargs['echo'] + '.xlsx', head=head)
        print(kwargs['echo'])

class Dictionary(object):

    def __init__(self, name):
        self.word2idx = {}
        with open(name, 'r', encoding='utf-8') as f:
            for index, line_temp in enumerate(f):
                line_temp = line_temp.strip()
                if not line_temp:
                    continue
                _word, _label = line_temp.split('\t')
                self.word2idx[_word] = _label

    def get_vector(self, text):
        seqs = []
        # text无值, 返回空list
        if not text:
            return []
        for word_temp in text:
            if self.word2idx.get(word_temp):
                seqs.append(int(self.word2idx[word_temp]))
            else:
                seqs.append(1)  # UNK
        return seqs

    def get_word_vector(self, single_word, **kwargs):
        result = kwargs.get('default_result', 0)
        if self.word2idx.get(single_word):
            result = str(self.word2idx[single_word])
        return result

class IntentionDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, name, sheet_index, start_line=1, shuffle = False):
        self.dic = Dictionary('../../data/dict/dept_classify_char_vocab.dic')
        data = get_train_test_data(name, sheet_index=sheet_index, start_line=start_line)
        data = [[temp[0], temp[1], label_num_intent_dict[temp[1]]] for temp in data if temp[1] != 'keyword']
        if shuffle:
            random.shuffle(data)
        if all_data_count is not None:
            data = data[: all_data_count]
        self.data = data

    def my_get(self, index):
        temp = self.data[index]
        input_ids = self.dic.get_vector(temp[0])
        label = temp[2]
        return input_ids, label, temp[0], len(input_ids)

    def __getitem__(self, index):
        temp = self.data[index]
        input_ids = self.dic.get_vector(temp[0])
        label = temp[2]
        return input_ids, label

    def __len__(self):
        return len(self.data)

def train_model():
    train_dataset = IntentionDataset(name=data_dir + 'intention.xls', sheet_index=0, shuffle=True)
    val_dataset = IntentionDataset(name=data_dir + 'intention.xls', sheet_index=1, shuffle=False)
    # weights = [4 if int(label) == 1 else 1 for _, _, label in train_dataset.data]
    # sampler = WeightedRandomSampler(weights, num_samples=8000, replacement=True)
    # sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset.data), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=0, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=custom_collate)
    print('init trainer....')
    trainer = Trainer()
    print('start train ....')
    trainer.train(train_loader, val_loader)
    trainer.t(name='%s.pkl' % t_name, echo='%s_last' % t_name)


def check_model():
    trainer = Trainer()
    # trainer.t(name='intention.pkl', echo='intention_last')
    trainer.t(name='%s.pkl' % t_name, echo=t_name)


if __name__ == '__main__':
    print('start ...')
    train_model()
    # check_model()
    print('ok')