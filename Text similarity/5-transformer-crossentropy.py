"""

@file  : 5-transformer-crossentropy.py

@author: xiaolu

@time  : 2019-09-25

"""
import tensorflow as tf
import re
import numpy as np
import pandas as pd
import collections
from unidecode import unidecode
from sklearn.model_selection import train_test_split


def position_encoding(inputs):
    T = tf.shape(inputs)[1]
    repr_dim = inputs.get_shape()[-1].value
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1])


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    return gamma * normalized + beta


def self_attention(inputs, is_training, num_units, num_heads=8, activation=None):
    T_q = T_k = tf.shape(inputs)[1]
    Q_K_V = tf.layers.dense(inputs, 3 * num_units, activation)
    Q, K, V = tf.split(Q_K_V, 3, -1)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), 0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), 0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), 0)
    align = tf.matmul(Q_, K_, transpose_b=True)
    align *= tf.rsqrt(tf.to_float(K_.get_shape()[-1].value))
    paddings = tf.fill(tf.shape(align), float('-inf'))
    lower_tri = tf.ones([T_q, T_k])
    lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
    masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
    align = tf.where(tf.equal(masks, 0), paddings, align)
    align = tf.nn.softmax(align)
    align = tf.layers.dropout(align, 0.1, training=is_training)
    x = tf.matmul(align, V_)
    x = tf.concat(tf.split(x, num_heads, axis=0), 2)
    x += inputs
    x = layer_norm(x)
    return x


def ffn(inputs, hidden_dim, activation=tf.nn.relu):
    x = tf.layers.conv1d(inputs, 4 * hidden_dim, 1, activation=activation)
    x = tf.layers.conv1d(x, hidden_dim, 1, activation=None)
    x += inputs
    x = layer_norm(x)
    return x


class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dropout, kernel_size=5):
        '''
        :param size_layer: 输出的维度
        :param num_layers: 几层
        :param embedded_size: 词嵌入的维度
        :param dict_size: 词表的大小
        :param learning_rate: 学习率
        :param dropout: dropout
        :param kernel_size: 卷积核大小
        '''
        def transformer_block(x, scope):
            x += position_encoding(x)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                for n in range(num_layers):
                    with tf.variable_scope('attn_%d' % i, reuse=tf.AUTO_REUSE):
                        x = self_attention(x, True, size_layer)
                    with tf.variable_scope('ffn_%d' % i, reuse=tf.AUTO_REUSE):
                        x = ffn(x, size_layer)

                with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
                    return tf.layers.dense(x, 2)[:, -1]

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        embedded_left = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        self.logits = transformer_block(embedded_left, 'left')
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y
            )
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(
            tf.argmax(self.logits, 1, output_type=tf.int32), self.Y
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def build_dataset(words, n_words):
    '''
    建立词典
    :param words:
    :param n_words:
    :return:
    '''
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
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


def str_idx(corpus, dic, maxlen, UNK=3):
    '''
    将语料转为对应的id序列
    :param corpus: 语料
    :param dic: 词典
    :param maxlen: 最大长度
    :param UNK: 不知道的标号
    :return:
    '''
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i][:maxlen][::-1]):
            val = dic[k] if k in dic else UNK
            X[i, -1 - no] = val
    return X


def cleaning(string):
    '''
    简单清洗语料
    :param string:
    :return:
    '''
    string = unidecode(string).replace('.', ' . ').replace(',', ' , ')
    string = re.sub('[^A-Za-z\- ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string).strip()
    return string.lower()


if __name__ == '__main__':
    df = pd.read_csv('./data/quora_duplicate_questions.tsv', delimiter='\t').dropna()
    print(df.head())

    # 取出两句话和标签
    left, right, label = df['question1'].tolist(), df['question2'].tolist(), df['is_duplicate'].tolist()

    # 看标签是不是只有两种
    print(np.unique(label, return_counts=True))   # (array([0, 1]), array([255024, 149263]))

    # 清洗语料　左右拼接到一块　中间用separator分隔开
    for i in range(len(left)):
        left[i] = cleaning(left[i])
        right[i] = cleaning(right[i])
        left[i] = left[i] + ' SEPARATOR ' + right[i]

    # 整理词表
    concat = ' '.join(left + right).split()
    vocabulary_size = len(list(set(concat)))
    data, count, vocab2id, id2vocab = build_dataset(concat, vocabulary_size)
    print("去重后的词的个数:", vocabulary_size)
    print("高频词:", count[4: 10])
    print("随机显示一些样本:", data[:10], [id2vocab[i] for i in data[:10]])

    # 定义超参数
    size_layer = 128
    num_layers = 4
    embedded_size = 128
    learning_rate = 1e-3
    maxlen = 50
    batch_size = 128
    dropout = 0.8

    # 将语料转为id序列
    vectors = str_idx(left, vocab2id, maxlen)

    # 切分数据集
    train_X, test_X, train_Y, test_Y = train_test_split(vectors, label, test_size=0.2)

    # 开始训练
    tf.reset_default_graph()
    sess = tf.Session()
    model = Model(size_layer, num_layers, embedded_size, len(vocab2id), learning_rate, dropout)
    sess.run(tf.global_variables_initializer())
    EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0
    while True:
        if CURRENT_CHECKPOINT == EARLY_STOPPING:
            print('break epoch: %d\n' % EPOCH)
            break
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        # train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []  # 收集每步的损失　
        for i in range(0, len(train_X), batch_size):
            batch_x = train_X[i: min(i + batch_size, train_X.shape[0])]
            batch_y = train_Y[i: min(i + batch_size, train_X.shape[0])]
            acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                    feed_dict={
                                        model.X: batch_x,
                                        model.Y: batch_y
                                    }
                                    )
            assert not np.isnan(loss)
            train_loss += loss
            train_acc += acc
            print("training--epoch: %d, step: %d, loss: %f, accuracy: %f" % (EPOCH, i // batch_size, loss, acc))

        # 测试集进行测试
        for i in range(0, len(test_X), batch_size):
            batch_x = test_X[i: min(i + batch_size, test_X.shape[0])]
            batch_y = test_Y[i: min(i + batch_size, test_X.shape[0])]
            acc, loss = sess.run([model.accuracy, model.cost],
                                 feed_dict={
                                     model.X: batch_x,
                                     model.Y: batch_y
                                 })
            test_loss += loss
            test_acc += acc
            print("testing--epoch: %d, step: %d, loss: %f, accuracy: %f" % (EPOCH, i // batch_size, loss, acc))

        train_loss /= (len(train_X) / batch_size)
        train_acc /= (len(train_X) / batch_size)
        test_loss /= (len(train_X) / batch_size)
        test_acc /= (len(train_X) / batch_size)

        if test_acc > CURRENT_ACC:
            # 测试集的准确率大于刚才的准确率 则继续进行训练
            print('epoch: %d, pass acc: %f, current acc: %f' % (EPOCH, CURRENT_ACC, test_acc))
            CURRENT_ACC = test_acc
            CURRENT_CHECKPOINT = 0
        else:
            CURRENT_CHECKPOINT += 1

        print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n' % (EPOCH, train_loss,
                                                                                                   train_acc, test_loss,
                                                                                                   test_acc))
