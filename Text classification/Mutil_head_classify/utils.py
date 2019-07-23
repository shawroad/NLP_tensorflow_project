"""

@file  : utils.py

@author: xiaolu

@time  : 2019-07-23

"""
import sklearn.datasets
import numpy as np
import re
import collections
import random
from sklearn import metrics
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')


def clearstring(string):
    '''
    清洗文本, 去除一些无用的字符
    :param string: 穿进来一条语料
    :return: 返回清洗过后的语料
    '''
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string if y.strip() not in english_stopwords]
    string = ' '.join(string)
    return string.lower()


def separate_dataset(trainset, ratio=0.5):
    '''
    切分数据集
    :param trainset: 传进来训练集
    :param ratio: 训练集和测试集的比例
    :return: 返回训练集的文本 与 标签
    '''
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        data_ = random.sample(data_, int(len(data_) * ratio))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


def build_dataset(words, n_words):
    '''
    建立词表
    :param words: 全体词
    :param n_words: 去重后的词
    :return: 返回词典
    '''
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
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


def str_idx(corpus, dic, maxlen, UNK = 3):
    '''
    将语料转为对应的id序列
    :param corpus: 语料
    :param dic: 词典
    :param maxlen: 最大长度
    :param UNK: 未知字符的标记
    :return: 将语料转为对应的id
    '''
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            X[i, -1 - no] = dic.get(k, UNK)
    return X
