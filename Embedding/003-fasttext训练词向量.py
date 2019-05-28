"""

@file   : 003-fasttext训练词向量.py

@author : xiaolu

@time1  : 2019-05-28

"""
import numpy as np
np.random.seed(1335)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.utils import np_utils
import random


def load_data(path, sent):
    # 加载数据集
    text = []
    label = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            text.append(line)
            label.append(sent)
    return text, label


def create_ngram_set(input_list, ngram_value=2):
    # create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    # 输出:[(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    result = []
    for i in range(len(input_list) - ngram_value + 1):
        temp = tuple(input_list[i: i + ngram_value])
        result.append(temp)
    # print(set(result))
    return set(result)


def add_ngram(sequences, token_indice, ngram_range=2):
    # 若: sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # 若: token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # 调用此函数: add_ngram(sequences, token_indice, ngram_range=2)
    # 最后的输出: [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range+1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i: i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


def extend_data(char_dic, data, labels):
    max_features = len(char_dic)
    # 将每句话转为id序列
    sentence2id = [[char_dic.get(word) for word in sen] for sen in data]
    ngram_range = 2

    if ngram_range > 1:
        ngram_set = set()
        for input_list in sentence2id:
            for i in range(2, ngram_range+1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
        # 这里是将咱们上面做的那n-gram对应一个数字。  这里直接从max_features  因为前面的数字词已经用了
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}  # N-gram => 数字
        # indice_token = {token_indice[k]: k for k in token_indice}  # 数字 => N-gram

    fea_dict = {**token_indice, **char_dic}  # 将两个字典合并

    sentence2id = add_ngram(sentence2id, fea_dict, ngram_range)

    # 将数据进行pad  同意将其pad成300
    X = sequence.pad_sequences(sentence2id, maxlen=300)
    labels = np_utils.to_categorical(labels)

    return X, labels, fea_dict


def build_model(fea_len, data, labels, char_dic):
    # 建立模型
    model = Sequential()
    # 特征嵌入  包括单个字 和 n_gram
    model.add(Embedding(fea_len, 200, input_length=300))
    # 我们增加 GlobalAveragePooling1D, 这将平均计算文档中所有词汇的的词嵌入
    model.add(GlobalAveragePooling1D())
    # 我们投射到单个单位的输出层上
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    model.summary()

    # 训练模型
    model.fit(data, labels, batch_size=30, epochs=3)

    # 把词嵌入那一层拿出来
    embedding_layer = model.get_layer("embedding_1")
    emb_wight = embedding_layer.get_weights()[0]

    def word2fea(word, char_dic):
        wordtuple = tuple(char_dic.get(i) for i in word)
        return wordtuple

    mather = word2fea("妈妈", char_dic)
    index = fea_dict.get(mather)
    mama = emb_wight[index]   # 获得那一层的词向量
    print(mama)


if __name__ == '__main__':
    neg = './data/neg.csv'
    pos = './data/pos.csv'
    n_label = 0  # 负例 标签为0
    p_label = 1  # 正例 标签为1
    data_neg, neg_label = load_data(neg, n_label)
    data_pos, pos_label = load_data(pos, p_label)
    # 样本 标签
    data_neg.extend(data_pos)
    neg_label.extend(pos_label)
    data = data_neg
    labels = neg_label

    # 将数据与标签打乱
    ind = [i for i in range(len(data))]
    random.shuffle(ind)
    shuffle_data = []
    shuffle_label = []
    for i in ind:
        shuffle_data.append(data[i])
        shuffle_label.append(labels[i])
    data = shuffle_data
    labels = shuffle_label


    # 字集合
    char_set = set(word for sen in data for word in sen)
    # 字和id的映射
    char_dic = {j: i+1 for i, j in enumerate(char_set)}
    char_dic['unk'] = 0   # 0表示不知

    # 数据增广
    data, labels, fea_dict = extend_data(char_dic, data, labels)

    fea_len = len(fea_dict)

    # 定义模型
    build_model(fea_len, data, labels, char_dic)




