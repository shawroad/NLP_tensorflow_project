"""

@file   : utils.py

@author : xiaolu

@time1  : 2019-05-24

"""
import sklearn.datasets
import numpy as np
import re
import collections
import random
from sklearn import metrics
from nltk.corpus import stopwords

# 获取英语的停用词
english_stopwords = stopwords.words('english')

def clearstring(string):
    # 数据预处理
    string = re.sub('[^A-Za-z0-9 ]+', '', string)  # 过滤无用字符
    string = string.split(' ')   # 分词
    string = filter(None, string)   # 过滤空字符
    # 去除停用词
    string = [y.strip() for y in string if y.strip() not in english_stopwords]
    string = ' '.join(string)
    return string.lower()


def separate_dataset(trainset, radio=0.5):
    # 切分数据集   整理好训练集
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        data_ = random.sample(data_, int(len(data_))*radio)   # 也就是从数据中抽取radio部分做为训练集
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])  # 每一条数据都去做预处理

        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])

    return datastring, datatarget


def build_dataset(words, n_words):
    # 将数据填充，进行id的映射
    # 起始填充0, 终止填充2, 长度不够的填充1, 不知的字符填充3
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reversed_dictionary


# str_idx(train_X[i:i+batch_size],dictionary,maxlen)
def str_idx(corpus, dic, maxlen, UNK=3):
    # corpus 一批数据     dic词典   maxlen 把数据都pad到这个长度
    X = np.zeros((len(corpus), maxlen), dtype=np.int32)
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            X[i, -1-no] = dic.get(k, UNK)   # UNK 表示获取不到这个词标记为3
    return X



