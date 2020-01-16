#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba

"""
jieba有3种分词模式
1.精确模式:试图将句子最精确地分开，适用文本分析
2.全匹配:可以成词地词语全扫描出来，有歧义
3.搜索引擎模式，在精确模式地基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
"""

sentence_1 = "我来到北京清华大学"
sentence_2 = "结婚的和尚未结婚的"

sentence = sentence_1

def precision(data):
    # 精确模式
    result = jieba.cut(sentence=data, cut_all=False)
    print('精确模式:\n%s\n' % '|'.join(result))

def full_match(data):
    # 完全匹配
    result = jieba.cut(sentence=data, cut_all=True)
    print('完全匹配模式:\n%s\n' % '|'.join(result))

def search_match(data):
    # 完全匹配
    result = jieba.cut_for_search(sentence=data)
    print('搜索引擎模式:\n%s\n' % '|'.join(result))


def check_load_dict():
    data = '新产品叫一网搜索'
    precision(data)
    print('load dict...')
    jieba.load_userdict('./data/jieba.dict')
    precision(data)



if __name__ == '__main__':
    precision(sentence)
    full_match(sentence)
    search_match(sentence)
    check_load_dict()