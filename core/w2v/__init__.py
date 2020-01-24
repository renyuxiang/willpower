"""

word2vec主要参数:
1.input_file：训练文件
2.output_file：输出向量文件
3.cbow：是否连续词袋模型
4.skip_gram：是否跳字模型
5.window：窗口大小
6.hs：1-采用Hierarchical Softmax模型， 2-负采样ns



组合：
1.cbow - hs
2.cbow - ns
3.skip_gram - hs
4.skip_gram - ns


需要攻克：
1.
"""