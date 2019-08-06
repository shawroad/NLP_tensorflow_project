"""

@file  : train.py

@author: xiaolu

@time  : 2019-08-05

"""
import numpy as np
from Model import Model
import tensorflow as tf


def load_data(path):
    corpus_1 = []
    corpus_2 = []
    labels = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            _, s1, s2, l1 = line.replace('\n', '').split('\t')
            corpus_1.append(s1)
            corpus_2.append(s2)
            labels.append(l1)
    return corpus_1, corpus_2, labels


def build_char(corpus_1, corpus_2):
    # 构建词表
    total_corpus = ''.join(corpus_1) + ''.join(corpus_2)
    char_ = list(set(list(total_corpus)))

    # 0 代表unk
    char2id = {'unk': 0}
    for i, c in enumerate(char_):
        char2id[c] = i+1

    # 再将语料中每句话转为数字序列
    corpus_1_num = []
    for c1 in corpus_1:
        temp = [char2id.get(c, 0) for c in list(c1)]
        corpus_1_num.append(temp)

    corpus_2_num = []
    for c2 in corpus_2:
        temp = [char2id.get(c, 0) for c in list(c2)]
        corpus_2_num.append(temp)

    # 找出最长序列, 然后将其他序列进行填充
    corpus_num = []
    corpus_num.extend(corpus_1_num)
    corpus_num.extend(corpus_2_num)
    maxlen = 0
    for c in corpus_num:
        if len(c) > maxlen:
            maxlen = len(c)

    # 进行padding
    for c1 in corpus_1_num:
        if len(c1) < maxlen:
            c1.extend((maxlen - len(c1)) * [0])
    for c2 in corpus_2_num:
        if len(c2) < maxlen:
            c2.extend((maxlen - len(c2)) * [0])
    return corpus_1_num, corpus_2_num, char2id


if __name__ == '__main__':
    path = './data/atec_nlp_sim_train_all.csv'
    corpus_1, corpus_2, labels = load_data(path)

    print('\n'.join(corpus_1[:4]))
    print('\n'.join(corpus_2[:4]))
    print(labels[:4])

    # 构建字表
    corpus_1_num, corpus_2_num, char2id = build_char(corpus_1, corpus_2)
    # corpus_1_num = np.array(corpus_1_num)
    # corpus_2_num = np.array(corpus_2_num)
    # print(corpus_1_num.shape)   # (102477, 112)
    #
    # def __init__(self, size_layer, num_layers, embedded_size,
    #                  dict_size, learning_rate, dropout, kernel_size=5):

    size_layer = 128
    num_layers = 3
    embedded_size = 128
    dict_size = len(char2id)
    learning_rate = 0.001
    dropout = 0.5

    model = Model(size_layer=size_layer,
                  num_layers=num_layers,
                  embedded_size=embedded_size,
                  dict_size=dict_size,
                  learning_rate=learning_rate,
                  )
    batch_size = 64
    num = len(corpus_1_num) // batch_size

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epoch = 10
    for e in range(epoch):
        print("当前epoch:", e)
        for i in range(num):
            batch_x1, batch_x2, batch_y = corpus_1_num[i: (i + 1)*batch_size], corpus_2_num[i: (i + 1)*batch_size], labels[i: (i + 1)*batch_size]
            _, loss, accuracy = sess.run([model.optimizer, model.cost, model.accuracy],
                                         feed_dict={model.X_left: batch_x1, model.X_right: batch_x2, model.Y: batch_y})

            average_loss = loss / batch_size
            average_acc = accuracy / batch_size
            print("当前第{}批数据, 损失:{}, 准确率:{}".format(i, average_loss, average_acc))
