import numpy as np
import jieba

"""
tf: 词汇在文档中出现的次数 / 文档中词汇总个数
idf: log(总文档数 / (包含该词汇的文档数 + 1) )

if_idf = tf * idf
"""
class TF_IDF_Model(object):
    def __init__(self, documents_list):
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 存储每个文本中每个词的词频
        self.tf = []
        # 存储每个词汇的逆文档频率
        self.idf = {}
        # 类初始化
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(self.documents_number / (value + 1))

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list

def check():
    document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                     "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                     "我在微信上被骗了，请问被骗多少钱才可以立案？",
                     "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                     "有人走私两万元，怎么处置他？",
                     "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]

    document_list = [list(jieba.cut(doc)) for doc in document_list]
    tf_idf_model = TF_IDF_Model(document_list)
    print(tf_idf_model.documents_list)
    print(tf_idf_model.documents_number)
    print(tf_idf_model.tf)
    print(tf_idf_model.idf)

    query = "走私了两万元，在法律上应该怎么量刑？"
    query = list(jieba.cut(query))
    scores = tf_idf_model.get_documents_score(query)
    print(scores)



if __name__ == '__main__':
    check()