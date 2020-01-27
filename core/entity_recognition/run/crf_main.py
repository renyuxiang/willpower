#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data import build_corpus
from core.entity_recognition.model.crf import CRFModel
from util.utils import save_model, load_model
from util.evaluating import Metrics

def crf():
    file_name = "./ryx_ckpts/crf.pkl"
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train", data_dir='../data')
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir='../data')
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir='../data')

    print("正在训练评估CRF模型...")
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, file_name)
    eval_model = load_model(file_name)
    # 评估模型
    print('eval test data')
    crf_eval(eval_model, test_word_lists, test_tag_lists)
    # print('eval dev data')
    # crf_eval(eval_model, dev_word_lists, dev_tag_lists)
    print('ok')


def crf_eval(eval_model, word_lists, tag_lists):
    remove_O = False
    pred_tag_lists = eval_model.test(word_lists)
    metrics = Metrics(tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

if __name__ == '__main__':
    crf()