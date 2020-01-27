#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from data import build_corpus
from util.utils import save_model, load_model, extend_maps
from util.evaluating import Metrics
from core.entity_recognition.model.bilstm_crf import BILSTM_Model

def lstm_eval(eval_model, word_lists, tag_lists, word2id, tag2id):
    remove_O = False
    pred_tag_lists, test_tag_lists = eval_model.test(
        word_lists, tag_lists, word2id, tag2id)
    metrics = Metrics(tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

def lstm(crf_mode=False):
    model_name = "bilstm_crf" if crf_mode else "bilstm"
    file_name = "../ckpts/%s.pkl" % model_name
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train", data_dir='../data')
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir='../data')
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir='../data')

    word2id, tag2id = extend_maps(word2id, tag2id, for_crf=False)
    print("正在训练评估lstm模型...")
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)

    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf_mode)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)
    save_model(bilstm_model, file_name)
    eval_model = load_model(file_name)
    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))
    print('eval test data')
    lstm_eval(eval_model, test_word_lists, test_tag_lists, word2id, tag2id)
    print('eval dev data')
    lstm_eval(eval_model, dev_word_lists, dev_tag_lists, word2id, tag2id)

if __name__ == '__main__':
    lstm()