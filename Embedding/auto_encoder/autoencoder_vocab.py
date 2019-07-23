"""

@file  : autoencoder_vocab.py

@author: xiaolu

@time  : 2019-07-23

"""
import sklearn.datasets
from sklearn.model_selection import train_test_split
import re
import tensorflow as tf
from sklearn import metrics
import numpy as np
import collections
import time
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns



def clearstring(string):
    '''
    清洗语料
    :param string: 一条语料
    :return: 清洗后的语料
    '''
    string = re.sub('[^A-Za-z ]+', '', string)
    string = string.split('\n')
    string = [y.strip() for y in filter(None, string)]
    string = (' '.join(string)).lower()
    return ' '.join([y.strip() for y in string.split()])


def separate_dataset(trainset):
    '''
    切分数据 返回 语料以及所对应的标签
    :param trainset: 数据
    :return: 返回语料, 标签
    '''
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


def build_dataset(words, n_words):
    '''
    建立词典
    :param words: 全部的词
    :param n_words: 去重后的词
    :return: 返回词表等
    '''
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words))
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


def str_idx(corpus, dic, maxlen, UNK=3):
    '''
    将语料转为对应的id序列
    :param corpus: 语料
    :param dic: 词典
    :param maxlen: 全部需要填充到这个维度
    :param UNK: 没有的词标记为unk
    :return: 返回的是语料转为id序列的numpy数组
    '''
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            try:
                X[i, -1 - no] = dic[k]
            except Exception as e:
                X[i, -1 - no] = UNK
    return X


# 加载数据集
dataset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')
dataset.data, dataset.target = separate_dataset(dataset)
_, dataset.data, _, dataset.target = train_test_split(dataset.data,
                                                      dataset.target,
                                                      test_size=0.03)
print(len(dataset.data))  # 320

concat = ' '.join(dataset.data).split()  # 全部的词
vocabulary_size = len(list(set(concat)))  # 去重后的词表
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print('vocab from size: %d' % vocabulary_size)   # 2328
print('Most common words', count[4:10])
print('Sample data', data[:10], [rev_dictionary[i] for i in data[:10]])

GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']


class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size,
                 dimension_output, learning_rate, seq_len):

        def cells(size, reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse)

        # define embedding
        self.X = tf.placeholder(tf.int32, [None, None])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], 0, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        # define MultiRNNCell
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer) for _ in range(num_layers)])
        outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

        # use all output then concat
        outputs = tf.reshape(outputs, [-1, seq_len * size_layer])

        # encoder
        W = tf.get_variable('w', shape=(size_layer*seq_len, dimension_output),
                            initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output), initializer=tf.zeros_initializer())

        self.logits = tf.nn.tanh(tf.matmul(outputs, W) + b)

        # decoder
        W_decoder = tf.get_variable('w_decoder', shape=(dimension_output, size_layer*seq_len),
                                    initializer=tf.orthogonal_initializer())
        b_decoder = tf.get_variable('b_decoder', shape=(size_layer*seq_len), initializer=tf.zeros_initializer())

        logits_decoder = tf.nn.tanh(tf.matmul(self.logits, W_decoder) + b_decoder)

        logits_decoder = tf.reshape(logits_decoder, [-1,seq_len, size_layer])

        with tf.variable_scope('decoder'):
            rnn_cells_decoder = tf.nn.rnn_cell.MultiRNNCell([cells(embedded_size) for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells_decoder, logits_decoder, dtype=tf.float32)

        decoder_dense = tf.layers.dense(outputs, dict_size)

        onehot_X = tf.one_hot(self.X, dict_size, axis=2, dtype=tf.float32)

        loss=tf.nn.softmax_cross_entropy_with_logits(labels=onehot_X, logits=decoder_dense)

        self.cost = tf.reduce_mean(loss)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)


size_layer = 256
num_layers = 2
embedded_size = 128
dimension_output = 100
learning_rate = 1e-2
maxlen = 50
batch_size = 32
epoch = 10


X = str_idx(dataset.data, dictionary, maxlen)
Y = dataset.target

tf.reset_default_graph()
sess = tf.Session()
model = Model(size_layer, num_layers, embedded_size, vocabulary_size+4,
              dimension_output, learning_rate, maxlen)
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    total_loss = 0
    for k in range(0, (X.shape[0]//batch_size)*batch_size, batch_size):
        loss, _ = sess.run([model.cost, model.optimizer], feed_dict={model.X: X[k:k+batch_size, :]})
        total_loss += loss
    total_loss /= (X.shape[0]//batch_size)
    print('epoch %d, avg loss %f' % (i+1, total_loss))

logits_test = sess.run(tf.nn.sigmoid(model.logits), feed_dict={model.X: X})
print(logits_test.shape)


# 将训练的此向量进行降维　降到2维 只是为了可视化
manifold_polarity = TSNE(n_components=2).fit_transform(logits_test)

sns.set()

ax_min = np.min(logits_test, 0)
ax_max = np.max(logits_test, 0)
ax_dist_sq = np.sum((ax_max-ax_min)**2)
plt.figure(figsize=(11, 11))
ax = plt.subplot(111)
shown_images = np.array([[1., 1.]])
colors = sns.color_palette(n_colors=len(dataset.target_names))
for i in range(manifold_polarity.shape[0]):
    dist = np.sum((manifold_polarity[i] - shown_images)**2, 1)
    if np.min(dist) < 3e-4*ax_dist_sq:
        continue
    shown_images = np.r_[shown_images, [manifold_polarity[i]]]
    plt.scatter(manifold_polarity[i,0],manifold_polarity[i,1],c=colors[Y[i]])
plt.legend(dataset.target_names)
plt.show()



