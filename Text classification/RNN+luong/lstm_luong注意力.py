"""

@file  : lstm_luong注意力.py

@author: xiaolu

@time  : 2019-07-23

"""
from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import sklearn
import numpy as np
from sklearn.metrics import classification_report


trainset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))

ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(trainset.data,
                                                                               trainset.target,
                                                                               ONEHOT, test_size=0.2)

concat = ' '.join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
# 构建词典
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print('vocab from size: %d' % vocabulary_size)
print('Most common words', count[4:10])
print('Sample data', data[:10], [rev_dictionary[i] for i in data[:10]])

GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']


class Model:
    def __init__(self, size_layer, num_layers, embedded_size,
                 dict_size, dimension_output, learning_rate):
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse)

        # 1. define placeholder
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        # 2.　define embedding
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        # 3. 加入Luong注意力
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=size_layer,
                                                                memory=encoder_embedded)

        rnn_cells = tf.contrib.seq2seq.AttentionWrapper(
            cell=tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)]),
            attention_mechanism=attention_mechanism,
            attention_layer_size=size_layer)

        # 单层动态rnn
        outputs, last_state = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

        # 输出
        W = tf.get_variable('w', shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs[:, -1], W) + b

        # 定义损失
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # 计算准确率
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


tf.reset_default_graph()
sess = tf.Session()
model = Model(size_layer,
              num_layers,
              embedded_size,
              vocabulary_size+4,
              dimension_output,
              learning_rate)

sess.run(tf.global_variables_initializer())


# 开始训练
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print('break epoch:%d\n' % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i:i + batch_size], dictionary, maxlen)
        acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                feed_dict={model.X: batch_x, model.Y: train_onehot[i:i + batch_size]})
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i:i + batch_size], dictionary, maxlen)
        acc, loss = sess.run([model.accuracy, model.cost],
                             feed_dict={model.X: batch_x, model.Y: test_onehot[i:i + batch_size]})
        test_loss += loss
        test_acc += acc

    train_loss /= (len(train_X) // batch_size)
    train_acc /= (len(train_X) // batch_size)
    test_loss /= (len(test_X) // batch_size)
    test_acc /= (len(test_X) // batch_size)

    if test_acc > CURRENT_ACC:
        print('epoch: %d, pass acc: %f, current acc: %f' % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print('time taken:', time.time() - lasttime)
    print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n' % (EPOCH, train_loss,
                                                                                               train_acc, test_loss,
                                                                                               test_acc))
    EPOCH += 1

logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names))
