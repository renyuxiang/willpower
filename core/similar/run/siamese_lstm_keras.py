#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Bidirectional, Lambda
import keras.backend as K
from willpower.util.common import w2excle
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model
from keras.models import model_from_json
import numpy as np
from willpower.util.common import Data2Vector
from sklearn.metrics import classification_report,roc_auc_score
from willpower.util.common import get_train_test_data

max_len = 100
n_words = 6000
embedding_dim = 100
hidden_dim = 128
epoch_value = 30
version = 1
file_path = 'siamese_keras_best.%s.weight' % version


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only,
                                                      mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)


def manhattandistance(l1, l2):
    return K.exp(-K.sum(K.abs(l1 - l2), axis=1, keepdims=True))


def network():
    ques1 = Input(shape=(max_len,), name='q1')
    ques2 = Input(shape=(max_len,), name='q2')

    base_model = Sequential()
    base_model.add(Embedding(n_words, embedding_dim))
    base_model.add(Bidirectional(
        LSTM(hidden_dim, dropout=0.2, recurrent_dropout=0.2,
             return_sequences=True, implementation=2)))
    base_model.add(Bidirectional(
        LSTM(hidden_dim, dropout=0.2, recurrent_dropout=0.2, implementation=2)))
    out1 = base_model(ques1)
    out2 = base_model(ques2)
    manhattan_dis = Lambda(lambda x: manhattandistance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([out1, out2])
    model = Model(inputs=[ques1, ques2], outputs=[manhattan_dis])
    print(model.summary())
    return model


def load_data(file):
    x1 = []
    x2 = []
    y = []
    with open(file=file, mode='r', encoding='utf-8') as f:
        for temp in f:
            data = temp.strip().split('\t')
            label = temp[0]
            q1 = [int(temp) for temp in data[1].split(',')]
            q2 = [int(temp) for temp in data[2].split(',')]
            x1.append(q1)
            x2.append(q2)
            y.append(int(label))
    assert len(x1) == len(x2)
    assert len(x1) == len(y)
    return x1, x2, y


def train():
    n_words = 6000
    classes = 2
    print('start loading data...')
    x1_train, x2_train, y_train = load_data('../data/train.txt')
    x1_val, x2_val, y_val = load_data('../data/val.txt')

    x1_train = pad_sequences(x1_train, maxlen=max_len)
    x2_train = pad_sequences(x2_train, maxlen=max_len)

    x1_val = pad_sequences(x1_val, maxlen=max_len)
    x2_val = pad_sequences(x2_val, maxlen=max_len)

    y_train = np.array(y_train)
    y_val = np.array(y_val)

    print(y_train.shape)
    # return

    available_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    source_model = network()
    model = multi_gpu_model(source_model, gpus=available_gpus)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    save_model_architecture(source_model, '.', 'siamese_keras', version)

    checkpoint = ParallelModelCheckpoint(
        source_model, file_path, monitor='val_acc', verbose=1, save_best_only=True,
        mode='auto', save_weights_only=True)
    print('start train...')
    cw = {0: 1, 1: 4}  # 类别权重
    model.fit([x1_train, x2_train], y_train, epochs=epoch_value,
              batch_size=1536, validation_data=([x1_val, x2_val], y_val), callbacks=[checkpoint],
              class_weight=cw, shuffle=True)
    save_model_weight(source_model, '.', 'siamese_keras', version)
    print('tran ok !!!')


def save_model_architecture(train_model, output_path, model_name, model_version):
    # 存储模型结构
    model_arch = '%s.%s.arch' % (model_name, model_version)
    model_arch_path = os.path.join(output_path, model_arch)
    json_string = train_model.to_json()
    open(model_arch_path, 'w').write(json_string)
    return


def save_model_weight(train_model, output_path, model_name, model_version):
    model_weight = '%s.%s.weight' % (model_name, model_version)
    model_weight_path = os.path.join(output_path, model_weight)
    train_model.save_weights(model_weight_path)
    return


def load_model(path, version):
    """加载已经存在的模型,其中version为版本号"""
    model_base_path = path
    model_path = model_base_path + '.' + '%s' + '.arch'
    model_arch = model_path % version
    model_weight_path = model_base_path + '.' + '%s' + '.weight'
    model_weight = model_weight_path % version
    model = model_from_json(open(model_arch).read(), custom_objects={'manhattandistance': manhattandistance})
    model.load_weights(model_weight, by_name=True)
    return model


def check(w_file):
    char_dic = Data2Vector.generate()
    from core.similar.data.deal_siamese_data import QuestionsDataset
    dataset = QuestionsDataset(sheet_index=2, dic=char_dic)
    result = []
    model = load_model('siamese_keras', 0)
    total = 0
    correct = 0
    y_trues = []
    y_hats = []
    y_preds = []
    for i, temp in enumerate(dataset.data):
        if i % 100 ==0:
            print(i)
        q1, q2, label = dataset.__getitem__(i)
        result_temp = [temp[0], temp[1], label]
        q1 = pad_sequences([q1], maxlen=max_len)
        q2 = pad_sequences([q2], maxlen=max_len)
        y_pred = model.predict([q1, q2]).item()
        y_hat = 0
        equal = 0
        total += 1
        if y_pred > 0.5:
            y_hat = 1
        if y_hat == label:
            equal = 1
            correct += 1
        result_temp.extend([y_hat, equal, y_pred])
        result.append(result_temp)
        y_trues.append(label)
        y_hats.append(y_hat)
        y_preds.append(y_pred)
    print(len(result))
    head = ['q1', 'q2', 'y', 'y_hat', 'equal', 'score']
    # metrics
    classification_report(y_trues, y_hats, target_names=['class0', 'class1'])
    roc_auc_score(y_trues, y_preds)
    w2excle(result, sheet_name='test_result', col=6, file_name= w_file, head=head)
    print('ok')


def check_result():
    # data = get_train_test_data('keras_result_1.xlsx', 1, start_line=1, cols=6)
    data = get_train_test_data('history/1/siamese_best.xlsx', 1, start_line=1, cols=6)
    y_trues = []
    y_preds = []
    y_hats = []
    for temp in data:
        y_trues.append(int(temp[2]))
        y_hats.append(int(temp[3]))
        y_preds.append(temp[5])
    print(classification_report(y_trues, y_hats, labels=[0, 1], target_names=['class0', 'class1']))
    print('auc:', roc_auc_score(y_trues, y_preds))


if __name__ == '__main__':
    # train()
    # load_data('../data/train.txt')
    # check(w_file='keras_result_1.xlsx')
    check_result()
    print('ok')