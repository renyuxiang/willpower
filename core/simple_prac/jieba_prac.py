#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import jieba
import jieba.posseg as pseg
from jieba import Tokenizer

"""
jieba有3种分词模式
1.精确模式:试图将句子最精确地分开，适用文本分析
2.全匹配:可以成词地词语全扫描出来，有歧义
3.搜索引擎模式，在精确模式地基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
"""

sentence_1 = "我来到北京清华大学"
sentence_2 = "结婚的和尚未结婚的"

sentence = sentence_1

def precision_match(data):
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


def pos():
    print('词性标注:')
    words = pseg.cut("我爱数据科学杂谈公众号")
    for w in words:
        print(w.word, w.flag)
    print('\n')


def check_load_dict():
    # 加载自定义词典
    data = '新产品叫一网搜索'
    precision_match(data)
    print('load dict...')
    jieba.load_userdict('./data/jieba.dict')
    precision_match(data)

def customize_dict():
    # 自定义词典
    small_jieba = Tokenizer('xx / dict.txt.small')  # 不传入任何参数，就使用默认的大字典文件
    result = small_jieba.lcut('我喜欢结巴分词！')  # 使用list(small_jieba.lcut())是一样的，但是看起来不专业
    print(list(result))

def cut_brackets():
    # 分词括号
    print("切除括号：")
    jieba.add_word('(泰诺)')
    _sen = '头痛怎么办，可以吃(泰诺)吗'
    precision_match(_sen)
    jieba.re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+  # &\._%（）\(\)]+)", re.U)
    precision_match(_sen)

if __name__ == '__main__':
    precision_match(sentence)
    full_match(sentence)
    search_match(sentence)
    check_load_dict()
    pos()
    cut_brackets()
