#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data import build_corpus
from ..model.hmm import HMM
from util.utils import save_model, load_model
from util.evaluating import Metrics

def hmm_eval(eval_model, word2id, tag2id, word_lists, tag_lists):
    remove_O = False
    pred_tag_lists = eval_model.test(word_lists, word2id, tag2id)
    metrics = Metrics(tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

def hmm():
    """
    https://spaces.ac.cn/archives/3922
    假设1：每个字的输出仅仅与当前字有关, P(o1,o2...on | k1,k2...kn) = P(o1 | k1) P(o2 | k2)... P(on | kn)
    o1 为 标签，k1为字, 估计P(o1 | k1)比较容易，代表 当前字为k1，输出标签为o1的概率。，只需要P(on | kn)最大化，
    该假设就是马尔可夫假设。  马尔科夫假设（Markov Assumption）：即下一个词的出现仅依赖于它前面的一个或几个词。

    以上假设没有考虑上下文，会出现不合理的情况，按照最大概率，会出现bbb的输出。由贝叶斯公式得到：
    P(o|k) = P(o, k) / P(k) = P(k|o)P(o) / P(k)
    由于k是给定的输入，那么P(k)就是常数，最大化P(o|k) 等价于最大化  P(k|o)P(o)，分别对P(k|o) 、 P(o)做马尔可夫假设，得到
    P(k|o) = P(k1|o1) P(k2|o2) ... P(kn|on)
    P(o) = P(o1) P(o2|o1) P(o3|o1,o2) ... P(on|o1,o2,...,o[n-1])
    再做1个马尔可夫假设,每个输出仅仅与上一个输出有关，那么
    P(o) = P(o1) P(o2|o1) P(o3|o2) ... P(on|o[n-1])
    此时 P(k|o)P(o) = P(k1|o1) * P(o2|o1) P(k2|o2) * P(o3|o2)... P(kn|on) * P(on|o[n-1])
    称P(kn|on)为发射概率， P(on|o[n-1])为转移概率, 用维特比算法解决
    """
    remove_O = False
    file_name = "./ryx_ckpts/hmm.pkl"
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train", data_dir='../data')
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    print("正在训练评估HMM模型...")
    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)

    save_model(hmm_model, file_name)

    eval_model = load_model(file_name)
    # 评估hmm模型
    print('eval dev data')
    hmm_eval(eval_model, word2id, tag2id, dev_word_lists, dev_tag_lists)
    print('eval test data')
    hmm_eval(eval_model, word2id, tag2id, test_word_lists, test_tag_lists)
    print('ok')