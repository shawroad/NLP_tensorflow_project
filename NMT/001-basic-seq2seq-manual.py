"""

@file  : 001-basic-seq2seq-manual.py

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


class Model:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size, learning_rate, batch_size):
        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        batch_size = tf.shape(self.X)[0]

        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))

        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        main = tf.strided_slice(self.X, [0, 0], [batch_size, -1], [1, 1])

        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        decoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, decoder_input)  # 解码的输出也是用的编码时的词嵌入
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])

        _, last_state = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded,
                                          sequence_length=self.X_seq_len,
                                          dtype=tf.float32)

        with tf.variable_scope("decoder"):
            rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells_dec, decoder_embedded,
                                           sequence_length=self.X_seq_len,
                                           initial_state=last_state,
                                           dtype=tf.float32)

        self.logits = tf.layers.dense(outputs, to_dict_size)
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                     targets=self.Y,
                                                     weights=masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        y_t = tf.argmax(self.logits, axis=2)
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


def pad_sentence_batch(sentence_batch, pad_int, maxlen):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
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

    size_layer = 256
    num_layers = 2
    embedded_size = 128
    learning_rate = 0.001
    batch_size = 16
    epoch = 20

    tf.reset_default_graph()
    sess = tf.Session()
    model = Model(size_layer, num_layers, embedded_size, len(vocab2id_from),
                  len(vocab2id_to), learning_rate, batch_size)

    sess.run(tf.global_variables_initializer())

    X = str_idx(text_from, vocab2id_from)
    Y = str_idx(text_to, vocab2id_to)

    maxlen_question = max([len(x) for x in X]) * 2
    maxlen_answer = max([len(y) for y in Y]) * 2

    for i in range(epoch):
        total_loss, total_accuracy = 0, 0
        X, Y = shuffle(X, Y)
        for k in range(0, len(text_to), batch_size):
            index = min(k + batch_size, len(text_to))
            batch_x, seq_x = pad_sentence_batch(X[k: index], PAD, maxlen_answer)
            batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD, maxlen_answer)
            predicted, accuracy, loss, _ = sess.run([tf.argmax(model.logits, 2),
                                                     model.accuracy,
                                                     model.cost, model.optimizer],
                                                    feed_dict={model.X: batch_x,
                                                               model.Y: batch_y,
                                                               model.X_seq_len: seq_x,
                                                               model.Y_seq_len: seq_y})
            total_loss += loss
            total_accuracy += accuracy
            print("当前步:{}, 损失:{}, 准确率:{}".format(k // batch_size, loss, accuracy))

        for i in range(len(batch_x)):
            print('row %d'%(i+1))
            print('QUESTION:', ' '.join([id2vocab_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]]))
            print('REAL ANSWER:', ' '.join([id2vocab_to[n] for n in batch_y[i] if n not in[0, 1, 2, 3]]))
            print('PREDICTED ANSWER:', ' '.join([id2vocab_to[n] for n in predicted[i] if n not in[0, 1, 2, 3]]), '\n')

        total_loss /= (len(text_to) / batch_size)
        total_accuracy /= (len(text_to) / batch_size)
        print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))

