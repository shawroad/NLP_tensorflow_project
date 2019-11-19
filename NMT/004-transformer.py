"""

@file  : 004-transformer.py

@author: xiaolu

@time  : 2019-09-04

"""
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import re
import time
import collections
import os


def layer_norm(inputs, epsilon=1e-8):
    '''
    层归一化
    :param inputs:
    :param epsilon:
    :return:
    '''
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


def multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads):
    '''
    多头注意力
    :param queries:
    :param keys:
    :param q_masks:
    :param k_masks:
    :param future_binding:
    :param num_units:
    :param num_heads:
    :return:
    '''
    T_q = tf.shape(queries)[1]
    T_k = tf.shape(keys)[1]

    Q = tf.layers.dense(queries, num_units, name='Q')
    K_V = tf.layers.dense(keys, 2 * num_units, name='K_V')
    K, V = tf.split(K_V, 2, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
    align = align / np.sqrt(K_.get_shape().as_list()[-1])

    paddings = tf.fill(tf.shape(align), float('-inf'))

    key_masks = k_masks
    key_masks = tf.tile(key_masks, [num_heads, 1])
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])
    align = tf.where(tf.equal(key_masks, 0), paddings, align)

    if future_binding:
        lower_tri = tf.ones([T_q, T_k])
        lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        align = tf.where(tf.equal(masks, 0), paddings, align)

    align = tf.nn.softmax(align)
    query_masks = tf.to_float(q_masks)
    query_masks = tf.tile(query_masks, [num_heads, 1])
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])
    align *= query_masks

    outputs = tf.matmul(align, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
    outputs += queries
    outputs = layer_norm(outputs)
    return outputs


def pointwise_feedforward(inputs, hidden_units, activation=None):
    '''
    位置向量
    :param inputs:
    :param hidden_units:
    :param activation:
    :return:
    '''
    outputs = tf.layers.dense(inputs, 4 * hidden_units, activation=activation)
    outputs = tf.layers.dense(outputs, hidden_units, activation=None)
    outputs += inputs
    outputs = layer_norm(outputs)
    return outputs


# def learned_position_encoding(inputs, mask, embed_dim):
#     '''
#     学习位置向量
#     :param inputs:
#     :param mask:
#     :param embed_dim:
#     :return:
#     '''
#     T = tf.shape(inputs)[1]
#     outputs = tf.range(tf.shape(inputs)[1])  # (T_q)
#     outputs = tf.expand_dims(outputs, 0)  # (1, T_q)
#     outputs = tf.tile(outputs, [tf.shape(inputs)[0], 1])  # (N, T_q)
#     outputs = embed_seq(outputs, T, embed_dim, zero_pad=False, scale=False)
#     return tf.expand_dims(tf.to_float(mask), -1) * outputs
#
#
def sinusoidal_position_encoding(inputs, mask, repr_dim):
    T = tf.shape(inputs)[1]
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1]) * tf.expand_dims(tf.to_float(mask), -1)


def label_smoothing(inputs, epsilon=0.1):
    C = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / C)


class Model:
    def __init__(self, size_layer, embedded_size, from_dict_size, to_dict_size, learning_rate,
                 num_blocks=2, num_heads=8, min_freq=50):
        # 1. define input
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]

        # 2. embedding
        encoder_embedding = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embedding = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))

        # 3. make decoder input. insert GO in 0 position
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)

        # 4. define forward net
        def forward(x, y):

            encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, x)
            en_masks = tf.sign(x)
            encoder_embedded += sinusoidal_position_encoding(x, en_masks, embedded_size)

            # add num block.  a block is consist of a multi_head and feedforward
            for i in range(num_blocks):
                with tf.variable_scope('encoder_self_attn_%d' % i, reuse=tf.AUTO_REUSE):
                    encoder_embedded = multihead_attn(queries=encoder_embedded,
                                                      keys=encoder_embedded,
                                                      q_masks=en_masks,
                                                      k_masks=en_masks,
                                                      future_binding=False,
                                                      num_units=size_layer,
                                                      num_heads=num_heads)

                with tf.variable_scope('encoder_feedforward_%d' % i, reuse=tf.AUTO_REUSE):
                    encoder_embedded = pointwise_feedforward(encoder_embedded,
                                                             embedded_size,
                                                             activation=tf.nn.relu)

            decoder_embedded = tf.nn.embedding_lookup(decoder_embedding, y)
            de_masks = tf.sign(y)
            decoder_embedded += sinusoidal_position_encoding(y, de_masks, embedded_size)

            for i in range(num_blocks):
                with tf.variable_scope('decoder_self_attn_%d' % i, reuse=tf.AUTO_REUSE):
                    decoder_embedded = multihead_attn(queries=decoder_embedded,
                                                      keys=decoder_embedded,
                                                      q_masks=de_masks,
                                                      k_masks=de_masks,
                                                      future_binding=True,
                                                      num_units=size_layer,
                                                      num_heads=num_heads)

                with tf.variable_scope('decoder_attn_%d' % i, reuse=tf.AUTO_REUSE):
                    decoder_embedded = multihead_attn(queries=decoder_embedded,
                                                      keys=encoder_embedded,
                                                      q_masks=de_masks,
                                                      k_masks=en_masks,
                                                      future_binding=False,
                                                      num_units=size_layer,
                                                      num_heads=num_heads)

                with tf.variable_scope('decoder_feedforward_%d' % i, reuse=tf.AUTO_REUSE):
                    decoder_embedded = pointwise_feedforward(decoder_embedded,
                                                             embedded_size,
                                                             activation=tf.nn.relu)

            return tf.layers.dense(decoder_embedded, to_dict_size, reuse=tf.AUTO_REUSE)

        self.training_logits = forward(self.X, decoder_input)

        def cond(i, y, temp):
            return i < 2 * tf.reduce_max(self.X_seq_len)

        def body(i, y, temp):
            logits = forward(self.X, y)
            ids = tf.argmax(logits, -1)[:, i]
            ids = tf.expand_dims(ids, -1)
            temp = tf.concat([temp[:, 1:], ids], -1)
            y = tf.concat([temp[:, -(i + 1):], temp[:, :-(i + 1)]], -1)
            y = tf.reshape(y, [tf.shape(temp)[0], 2 * tf.reduce_max(self.X_seq_len)])
            i += 1
            return i, y, temp

        target = tf.fill([batch_size, 2 * tf.reduce_max(self.X_seq_len)], GO)
        target = tf.cast(target, tf.int64)
        self.target = target

        _, self.predicting_ids, _ = tf.while_loop(cond, body,
                                                  [tf.constant(0), target, target])

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                     targets=self.Y,
                                                     weights=masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def build_dataset(words, n_words, atleast=1):
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)

    vocab2id = dict()
    for word, _ in count:
        vocab2id[word] = len(vocab2id)

    data = list()
    unk_count = 0
    for word in words:
        index = vocab2id.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count

    id2vocab = dict(zip(vocab2id.values(), vocab2id.keys()))
    return data, count, vocab2id, id2vocab


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


def str_idx(corpus, dic):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k,UNK))
        X.append(ints)
    return X


if __name__ == '__main__':
    with open('./data/english-train', 'r') as fopen:
        text_from = fopen.read().lower().split('\n')[:-1]
    with open('./data/vietnam-train', 'r') as fopen:
        text_to = fopen.read().lower().split('\n')[:-1]

    print("len from: %d, len to: %d"%(len(text_from), len(text_to)))   # 500, 500

    # 整理的是英文数据
    concat_from = ' '.join(text_from).split()
    vocabulary_size_from = len(list(set(concat_from)))  # 英语去重后的所有词
    data_from, count_from, vocab2id_from, id2vocab_from = build_dataset(concat_from, vocabulary_size_from)
    print('vocab from size: %d' % vocabulary_size_from)
    print('Most common words', count_from[4:10])
    print('Sample data', data_from[:10], [id2vocab_from[i] for i in data_from[:10]])

    # 整理vietnam语数据
    concat_to = ' '.join(text_to).split()
    vocabulary_size_to = len(list(set(concat_to)))
    data_to, count_to, vocab2id_to, id2vocab_to = build_dataset(concat_to, vocabulary_size_to)
    print('vocab to size: %d' % vocabulary_size_to)
    print('Most common words', count_to[4:10])
    print('Sample data', data_to[:10], [id2vocab_to[i] for i in data_to[:10]])

    GO = vocab2id_from['GO']
    PAD = vocab2id_from['PAD']
    EOS = vocab2id_from['EOS']
    UNK = vocab2id_from['UNK']

    # 给越南语每条语句最后加EOS
    for i in range(len(text_to)):
        text_to[i] += ' EOS'

    embedded_size = 256
    learning_rate = 0.001
    batch_size = 16
    epoch = 20

    tf.reset_default_graph()
    sess = tf.Session()

    model = Model(embedded_size, embedded_size, len(vocab2id_from),
                    len(vocab2id_to), learning_rate)

    sess.run(tf.global_variables_initializer())

    X = str_idx(text_from, vocab2id_from)
    Y = str_idx(text_to, vocab2id_to)

    maxlen_question = max([len(x) for x in X]) * 2
    maxlen_answer = max([len(y) for y in Y]) * 2

    for i in range(epoch):
        total_loss, total_accuracy = 0, 0
        for k in range(0, len(text_to), batch_size):
            index = min(k + batch_size, len(text_to))
            batch_x, seq_x = pad_sentence_batch(X[k: index], PAD)
            batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD)
            predicted, accuracy, loss, _ = sess.run([model.predicting_ids,
                                                     model.accuracy, model.cost, model.optimizer],
                                                    feed_dict={model.X: batch_x,
                                                               model.Y: batch_y})
            total_loss += loss
            total_accuracy += accuracy
            print("当前步:{}, 损失:{}, 准确率:{}".format(k // batch_size, loss, accuracy))

        total_loss /= (len(text_to) / batch_size)
        total_accuracy /= (len(text_to) / batch_size)
        print('epoch: %d, avg loss: %f, avg accuracy: %f' % (i + 1, total_loss, total_accuracy))