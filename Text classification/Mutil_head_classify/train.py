"""

@file  : train.py

@author: xiaolu

@time  : 2019-07-23

"""
from utils import *
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


trainset = datasets.load_files(container_path='data', encoding='UTF-8')
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))


# 将标签转为one_hot编码  记住: 这里只需要将训练集的标签转为one_hot编码
ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(trainset.data, trainset.target,
                                                                               ONEHOT, test_size=0.2)
print(len(train_X))   # 8529
print(len(test_X))   # 2133
print(len(train_onehot))  # 8529


# 构建词表
concat = ' '.join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, vocb2id, id2vocb = build_dataset(concat, vocabulary_size)

print('词表的大小: %d'%(vocabulary_size))
print('最常用的几个词: ', count[4:10])
print('打印几个样本看看: ', data[:10], [id2vocb[i] for i in data[:10]])

GO = vocb2id['GO']
PAD = vocb2id['PAD']
EOS = vocb2id['EOS']
UNK = vocb2id['UNK']
print(GO, PAD, EOS, UNK)


def embed_seq(inputs, vocab_size=None, embed_dim=None, zero_pad=False, scale=False):
    lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, embed_dim])
    if zero_pad:
        lookup_table = tf.concat((tf.zeros([1, embed_dim]), lookup_table[1:, :]), axis=0)
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)
    if scale:
        outputs = outputs * (embed_dim ** 0.5)
    return outputs


def learned_positional_encoding(inputs, embed_dim, zero_pad=False, scale=False):
    T = inputs.get_shape().as_list()[1]
    outputs = tf.range(T)
    outputs = tf.expand_dims(outputs, 0)
    outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])
    return embed_seq(outputs, T, embed_dim, zero_pad=zero_pad, scale=scale)


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    return gamma * normalized + beta


def pointwise_feedforward(inputs, num_units=[None, None], activation=None):
    outputs = tf.layers.conv1d(inputs, num_units[0], kernel_size=1, activation=activation)
    outputs = tf.layers.conv1d(outputs, num_units[1], kernel_size=1, activation=None)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


class Model:
    def __init__(self, dict_size, dimension_input, dimension_output, seq_len,
                 learning_rate, num_heads=8, attn_windows=range(1, 6)):
        self.size_layer = dimension_input
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.X = tf.placeholder(tf.int32, [None, seq_len])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, dimension_input], -1, 1))
        feed = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        for i, win_size in enumerate(attn_windows):
            with tf.variable_scope('attn_masked_window_%d' % win_size):
                feed = self.multihead_attn(feed, self.window_mask(win_size))
        feed += learned_positional_encoding(feed, dimension_input)
        with tf.variable_scope('multihead'):
            feed = self.multihead_attn(feed, None)
        with tf.variable_scope('pointwise'):
            feed = pointwise_feedforward(feed, num_units=[4 * dimension_input,
                                                          dimension_input], activation=tf.nn.relu)
        self.logits = tf.layers.dense(feed, dimension_output)[:, -1]
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def multihead_attn(self, inputs, masks):
        T_q = T_k = inputs.get_shape().as_list()[1]
        Q_K_V = tf.layers.dense(inputs, 3 * self.size_layer, tf.nn.relu)
        Q, K, V = tf.split(Q_K_V, 3, -1)
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
        align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        align = align / np.sqrt(K_.get_shape().as_list()[-1])
        if masks is not None:
            paddings = tf.fill(tf.shape(align), float('-inf'))
            align = tf.where(tf.equal(masks, 0), paddings, align)
        align = tf.nn.softmax(align)
        outputs = tf.matmul(align, V_)
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
        outputs += inputs
        return layer_norm(outputs)

    def window_mask(self, h_w):
        masks = np.zeros([self.seq_len, self.seq_len])
        for i in range(self.seq_len):
            if i < h_w:
                masks[i, :i + h_w + 1] = 1.
            elif i > self.seq_len - h_w - 1:
                masks[i, i - h_w:] = 1.
            else:
                masks[i, i - h_w:i + h_w + 1] = 1.
        masks = tf.convert_to_tensor(masks)
        return tf.tile(tf.expand_dims(masks, 0), [tf.shape(self.X)[0] * self.num_heads, 1, 1])


embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128

tf.reset_default_graph()
sess = tf.Session()
model = Model(vocabulary_size+4, embedded_size, dimension_output, maxlen, learning_rate)
sess.run(tf.global_variables_initializer())


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print('break epoch:%d\n' % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i:i + batch_size], vocb2id, maxlen)
        acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                feed_dict={model.X: batch_x, model.Y: train_onehot[i:i + batch_size]})
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i:i + batch_size], vocb2id, maxlen)
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


