"""

@file  : 2-dilated-cnn-contrastive.py

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
    '''
    位置编码
    :param inputs:
    :return:
    '''
    T = tf.shape(inputs)[1]
    repr_dim = inputs.get_shape()[-1].value
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1])


def layer_norm(inputs, epsilon=1e-8):
    '''
    层归一化
    :param inputs:
    :param epsilon:
    :return:
    '''
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))  # 标准化
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    return gamma * normalized + beta


def cnn_block(x, dilation_rate, pad_sz, hidden_dim, kernel_size):
    '''
    空洞卷积块
    :param x: 输入
    :param dilation_rate: 空洞率
    :param pad_sz: 填充大小
    :param hidden_dim: 卷积核个数
    :param kernel_size: 卷积核大小
    :return:
    '''
    x = layer_norm(x)
    pad = tf.zeros([tf.shape(x)[0], pad_sz, hidden_dim])
    x = tf.layers.conv1d(inputs=tf.concat([pad, x, pad], 1),
                         filters=hidden_dim,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate
                         )
    x = x[:, :-pad_sz, :]
    x = tf.nn.relu(x)
    return x


# 定义模型
class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dropout, kernel_size=5):
        '''
        :param size_layer:
        :param num_layers:
        :param embedded_size:
        :param dict_size:
        :param learning_rate:
        :param dropout:
        :param kernel_size:
        '''
        def cnn(x, scope):
            # 加入位置向量
            x += position_encoding(x)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                # 来几层空洞卷积
                for n in range(num_layers):
                    dilation_rate = 2 ** n   # 第一层空洞率: 2, 第二层空洞: 4
                    pad_sz = (kernel_size - 1) * dilation_rate
                    with tf.variable_scope('block_%d' % i, reuse=tf.AUTO_REUSE):
                        x += cnn_block(x, dilation_rate, pad_sz, size_layer, kernel_size)

                with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
                    return tf.layers.dense(x, size_layer)[:, -1]

        # 定义输入
        self.X_left = tf.placeholder(tf.int32, [None, None])
        self.X_right = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None])
        self.batch_size = tf.shape(self.X_left)[0]

        # 对左, 右进行词嵌入
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        embedded_left = tf.nn.embedding_lookup(encoder_embeddings, self.X_left)
        embedded_right = tf.nn.embedding_lookup(encoder_embeddings, self.X_right)

        def contrastive_loss(y, d):
            tmp = y * tf.square(d)
            tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
            return tf.reduce_sum(tmp + tmp2) / tf.cast(self.batch_size, tf.float32) / 2

        self.output_left = cnn(embedded_left, 'left')
        self.output_right = cnn(embedded_right, 'right')

        print(self.output_left, self.output_right)

        self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.output_left, self.output_right)),
                                              1, keep_dims=True))
        self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.output_left),
                                                                           1, keep_dims=True)),
                                                     tf.sqrt(tf.reduce_sum(tf.square(self.output_right),
                                                                           1, keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1])
        self.cost = contrastive_loss(self.Y, self.distance)

        self.temp_sim = tf.subtract(tf.ones_like(self.distance),
                                    tf.rint(self.distance))
        correct_predictions = tf.equal(self.temp_sim, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


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

    # 清洗语料
    for i in range(len(left)):
        left[i] = cleaning(left[i])
        right[i] = cleaning(right[i])

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
    vectors_left = str_idx(left, vocab2id, maxlen)
    vectors_right = str_idx(right, vocab2id, maxlen)

    # 切分数据集
    train_X_left, test_X_left, train_X_right, test_X_right, train_Y, test_Y = train_test_split(vectors_left,
                                                                                               vectors_right,
                                                                                               label,
                                                                                               test_size=0.2)

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
        for i in range(0, len(train_X_left), batch_size):
            batch_x_left = train_X_left[i: min(i + batch_size, train_X_left.shape[0])]
            batch_x_right = train_X_right[i: min(i + batch_size, train_X_left.shape[0])]
            batch_y = train_Y[i: min(i + batch_size, train_X_left.shape[0])]
            acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                    feed_dict={
                                        model.X_left: batch_x_left,
                                        model.X_right: batch_x_right,
                                        model.Y: batch_y
                                    }
                                    )
            assert not np.isnan(loss)
            train_loss += loss
            train_acc += acc
            print("training--epoch: %d, step: %d, loss: %f, accuracy: %f" % (EPOCH, i // batch_size, loss, acc))

        # 测试集进行测试
        for i in range(0, len(test_X_left), batch_size):
            batch_x_left = test_X_left[i: min(i + batch_size, test_X_left.shape[0])]
            batch_x_right = test_X_right[i: min(i + batch_size, test_X_left.shape[0])]
            batch_y = test_Y[i: min(i + batch_size, test_X_left.shape[0])]
            acc, loss = sess.run([model.accuracy, model.cost],
                                 feed_dict={
                                     model.X_left: batch_x_left,
                                     model.X_right: batch_x_right,
                                     model.Y: batch_y
                                 })
            test_loss += loss
            test_acc += acc
            print("testing--epoch: %d, step: %d, loss: %f, accuracy: %f" % (EPOCH, i // batch_size, loss, acc))

        train_loss /= (len(train_X_left) / batch_size)
        train_acc /= (len(train_X_left) / batch_size)
        test_loss /= (len(test_X_left) / batch_size)
        test_acc /= (len(test_X_left) / batch_size)

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

    left = str_idx(['a person is outdoors, on a horse.'], vocab2id, maxlen)
    right = str_idx(['a person on a horse jumps over a broken down airplane.'], vocab2id, maxlen)
    sess.run([model.temp_sim, 1-model.distance],
             feed_dict={
                 model.X_left: left,
                 model.X_right: right})
