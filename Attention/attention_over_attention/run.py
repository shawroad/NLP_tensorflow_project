"""

@file  : run.py

@author: xiaolu

@time  : 2019-07-17

"""
import glob
import re
import numpy as np
from attn_over_attn import Model
from keras.utils import to_categorical
import tensorflow as tf


def load_data(path, label):
    with open(path, 'r', encoding='utf8') as f:
        # 制作标签
        if path[-3:] == 'pos':
            sign = 0
        else:
            sign = 1

        lines = f.readlines()
        data = []
        label = []
        for line in lines:
            line = line.replace('\n', '')
            data.append(line)
            label.append(sign)
    return data, label


def clean_data(string):
    # 简单对数据进行清洗
    # 对数据清洗
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def process_data(datas):
    total_str = ' '.join(datas)

    # 1.词表
    vocab = list(set(total_str.split(' ')))

    # 2.词与id映射
    # PAD: 0, UNK: 1
    vocab2id = {}
    vocab2id['PAD'] = 0
    vocab2id['UNK'] = 1
    for i, v in enumerate(vocab):
        vocab2id[v] = i+2
    # print(len(vocab2id))  # 18766
    # print(vocab2id)

    length = 0
    i = 0
    split_sentence = []
    for d in datas:
        temp = d.split(' ')
        split_sentence.append(temp)
        length += len(temp)
        i += 1
    average_len = length // i
    # print("平均长度:", average)   # 平均长度: 20
    # print(split_sentence[0])

    final_data = []
    for d in split_sentence:
        temp = [vocab2id.get(i, 1) for i in d]
        if len(temp) > average_len:
            temp = temp[:average_len]
        else:
            temp += [0] * (average_len - len(temp))
        final_data.append(temp)

    return final_data, average_len, vocab2id


if __name__ == '__main__':
    path_list = glob.glob('./data/*')
    # print(path_list)   # ['./data/rt-polarity.pos', './data/rt-polarity.neg']
    label = {'pos': 0, 'neg': 1}
    datas, labels = [], []
    for path in path_list:
        data, label = load_data(path, label)
        datas.extend(data)
        labels.extend(label)
    labels = np.array(labels)

    # 对数据进行简单的清洗
    data_temp = []
    for data in datas:
        temp = clean_data(data)
        data_temp.append(temp)
    datas = data_temp
    del data_temp

    # 建立词表　并将字符串转为数字序列, 最后进行padding
    final_data, average_len, vocab2id = process_data(datas)
    X = np.array(final_data)
    Y = to_categorical(labels)

    # 打乱数据
    index = np.random.permutation(X.shape[0])
    data_ = []
    label_ = []
    for i in index:
        data_.append(X[i])
        label_.append(Y[i])

    X, Y = data_, label_

    num_layers = 2
    embedded_size = 128
    learning_rate = 1e-3
    maxlen = 20
    batch_size = 128
    size_layer = 128
    dict_size = len(vocab2id)
    dimension_output = 2

    model = Model(size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen)
    total_step = len(final_data) // batch_size
    epochs = 10
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("Epoch:{}".format(epoch))
        for step in range(total_step):
            batch_x, batch_y = X[step*batch_size: (step+1)*batch_size], Y[step*batch_size: (step+1)*batch_size]
            _, loss, acc = sess.run([model.optimizer, model.cost, model.accuracy], feed_dict={
                model.X: batch_x, model.Y: batch_y
            })
            print("step: {}, loss: {}, accuracy: {}".format(step, loss, acc))
