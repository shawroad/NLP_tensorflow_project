"""

@file  : 003-dilated-seq2seq-self-attention.py

@author: xiaolu

@time  : 2019-09-26

"""
import json
import numpy as np
import tensorflow as tf
import collections
from sklearn.model_selection import train_test_split
from tensor2tensor.utils import beam_search
import re
from unidecode import unidecode
import random
from sklearn.utils import shuffle
import time


def embed_seq(x, vocab_sz, embed_dim, name, zero_pad=True):
    '''
    词嵌入
    :param x:
    :param vocab_sz:
    :param embed_dim:
    :param name:
    :param zero_pad:
    :return:
    '''
    embedding = tf.get_variable(name, [vocab_sz, embed_dim])
    if zero_pad:
        embedding = tf.concat([tf.zeros([1, embed_dim]), embedding[1:, :]], 0)
    x = tf.nn.embedding_lookup(embedding, x)
    return x


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
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    return gamma * normalized + beta


def cnn_block(x, dilation_rate, pad_sz, hidden_dim, kernel_size):
    '''
    卷积块
    :param x:
    :param dilation_rate:
    :param pad_sz:
    :param hidden_dim:
    :param kernel_size:
    :return:
    '''
    x = layer_norm(x)
    pad = tf.zeros([tf.shape(x)[0], pad_sz, hidden_dim])
    x = tf.layers.conv1d(inputs=tf.concat([pad, x, pad], 1),
                         filters=hidden_dim,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate)
    x = x[:, :-pad_sz, :]
    x = tf.nn.relu(x)
    return x


def Attention(Q, inputs, num_units, num_heads=8, activation=None):
    '''
    self-attention
    :param Q:
    :param inputs:
    :param num_units:
    :param num_heads:
    :param activation:
    :return:
    '''
    inputs = tf.layers.dropout(inputs, 0.1, training=True)
    T_q = tf.shape(Q)[1]
    T_k = tf.shape(inputs)[1]
    K_V = tf.layers.dense(inputs, 2 * num_units, activation)
    K, V = tf.split(K_V, 2, -1)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), 0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), 0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), 0)
    p_gen = tf.layers.dense(K * V, 1)
    p_gen = tf.sigmoid(p_gen)
    align = tf.matmul(Q_, K_, transpose_b=True)
    align *= tf.rsqrt(tf.to_float(K_.get_shape()[-1].value))
    paddings = tf.fill(tf.shape(align), float('-inf'))
    lower_tri = tf.ones([T_q, T_k])
    lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
    masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
    align = tf.where(tf.equal(masks, 0), paddings, align)
    align = tf.nn.softmax(align)
    alignments = tf.transpose(align, [0, 2, 1])
    x = tf.matmul(align, V_)
    x = tf.concat(tf.split(x, num_heads, axis=0), 2)
    x += Q
    x = layer_norm(x)
    return x, alignments, p_gen


class Summarization:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, kernel_size=2, n_attn_heads=16):

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]
        self.batch_size = batch_size
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)

        self.embedding = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.size_layer = size_layer
        self.n_attn_heads = n_attn_heads
        self.dict_size = dict_size

        self.training_logits = self.forward(self.X, decoder_input)

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

    def forward(self, x, y, reuse=False):
        with tf.variable_scope('forward', reuse=reuse):
            with tf.variable_scope('forward', reuse=reuse):
                encoder_embedded = tf.nn.embedding_lookup(self.embedding, x)
                decoder_embedded = tf.nn.embedding_lookup(self.embedding, y)
                encoder_embedded += position_encoding(encoder_embedded)

                for i in range(self.num_layers):
                    dilation_rate = 2 ** i
                    pad_sz = (self.kernel_size - 1) * dilation_rate
                    with tf.variable_scope('block_%d' % i, reuse=reuse):
                        encoder_embedded += cnn_block(encoder_embedded, dilation_rate,
                                                      pad_sz, self.size_layer, self.kernel_size)

                g = tf.identity(decoder_embedded)
                for i in range(self.num_layers):
                    dilation_rate = 2 ** i
                    pad_sz = (self.kernel_size - 1) * dilation_rate
                    with tf.variable_scope('decode_%d' % i, reuse=reuse):
                        attn_res = h = cnn_block(decoder_embedded, dilation_rate,
                                                 pad_sz, self.size_layer, self.kernel_size)
                        with tf.variable_scope('attention_%d' % i, reuse=reuse):
                            h, alignment, p_gen = Attention(attn_res, encoder_embedded, self.size_layer)

                        vocab_dist = tf.nn.softmax(encoder_embedded) * p_gen
                        alignments = alignment[0] * (1 - p_gen)
                        encoder_embedded += vocab_dist
                        decoder_embedded += h

                return tf.layers.dense(decoder_embedded, self.dict_size)


def textcleaning(string):
    '''
    对文本进行清洗
    :param string: 
    :return: 
    '''
    string = unidecode(string).replace('.', '. ').replace(',', ' , ')
    string = re.sub('[^\'"A-Za-z\- ]+', ' ', string)
    string = re.sub(r'[ ]+', ' ', string.lower()).strip().split()
    if len(string) > max_len:
        string = random.sample(string, max_len)
    return ' '.join(string)


def build_dataset(words, n_words):
    '''
    建立词典
    :param words: 
    :param n_words: 
    :return: 
    '''
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words))
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
    reversed_vocab2id = dict(zip(vocab2id.values(), vocab2id.keys()))
    return data, count, vocab2id, reversed_vocab2id


def str_idx(corpus, dic, UNK=3):
    '''
    将文本转为id序列
    :param corpus:
    :param dic:
    :param UNK:
    :return:
    '''
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


def pad_sentence_batch(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


def beam_search_decoding(length=20, beam_width=5):
    initial_ids = tf.fill([model.batch_size], GO)

    def symbols_to_logits(ids):
        x = tf.contrib.seq2seq.tile_batch(model.X, beam_width)
        logits = model.forward(x, ids, reuse=True)
        return logits[:, tf.shape(ids)[1] - 1, :]

    final_ids, final_probs, _ = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_width,
        length,
        len(vocab2id),
        0.0,
        eos_id=EOS)

    return final_ids


if __name__ == '__main__':
    # 1. 加载语料
    with open('./dataset/ctexts.json', 'r') as fopen:
        ctexts = json.load(fopen)
    
    with open('./dataset/headlines.json', 'r') as fopen:
        headlines = json.load(fopen)

    max_len = 500
    
    # 2. 清洗文本
    h, c = [], []
    for i in range(len(ctexts)):
        try:
            c.append(textcleaning(ctexts[i]))
            h.append(textcleaning(headlines[i]))
        except:
            pass
        
    concat = ' '.join(c).split()
    vocabulary_size = len(list(set(concat)))
    data, count, vocab2id, id2vocab = build_dataset(concat, vocabulary_size)
    print('vocab from size: %d' % vocabulary_size)
    print('Most common words', count[4:10])
    print('Sample data', data[:10], [id2vocab[i] for i in data[:10]])
    print('filtered vocab size:', len(vocab2id))
    print("% of vocab used: {}%".format(round(len(vocab2id)/vocabulary_size, 4)*100))

    for i in range(len(h)):
        h[i] = h[i] + ' EOS'

    GO = vocab2id['GO']
    PAD = vocab2id['PAD']
    EOS = vocab2id['EOS']
    UNK = vocab2id['UNK']

    X = str_idx(c, vocab2id)
    Y = str_idx(h, vocab2id)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)

    size_layer = 128
    num_layers = 4
    embedded_size = 128
    learning_rate = 1e-3
    batch_size = 8
    epoch = 20

    tf.reset_default_graph()
    sess = tf.Session()
    model = Summarization(size_layer, num_layers, embedded_size, len(vocab2id), learning_rate)
    model.generate = beam_search_decoding()
    sess.run(tf.global_variables_initializer())

    for EPOCH in range(10):
        lasttime = time.time()
        total_loss, total_accuracy, total_loss_test, total_accuracy_test = 0, 0, 0, 0
        train_X, train_Y = shuffle(train_X, train_Y)
        test_X, test_Y = shuffle(test_X, test_Y)
        for k in range(0, len(train_X), batch_size):
            batch_x, _ = pad_sentence_batch(train_X[k: min(k + batch_size, len(train_X))], PAD)
            batch_y, _ = pad_sentence_batch(train_Y[k: min(k + batch_size, len(train_X))], PAD)
            acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                    feed_dict={model.X: batch_x,
                                               model.Y: batch_y})
            total_loss += loss
            total_accuracy += acc
            print("training--epoch: %d, step: %d, loss: %f, accuracy: %f" % (EPOCH, k // batch_size, loss, acc))

        for k in range(0, len(test_X), batch_size):
            batch_x, _ = pad_sentence_batch(test_X[k: min(k + batch_size, len(test_X))], PAD)
            batch_y, _ = pad_sentence_batch(test_Y[k: min(k + batch_size, len(test_X))], PAD)
            acc, loss = sess.run([model.accuracy, model.cost],
                                 feed_dict={
                                     model.X: batch_x,
                                     model.Y: batch_y
                                 }
                                 )
            total_loss_test += loss
            total_accuracy_test += acc
            print("testing--epoch: %d, step: %d, loss: %f, accuracy: %f" % (EPOCH, k // batch_size, loss, acc))

        total_loss /= (len(train_X) / batch_size)
        total_accuracy /= (len(train_X) / batch_size)
        total_loss_test /= (len(test_X) / batch_size)
        total_accuracy_test /= (len(test_X) / batch_size)

        print('epoch: %d, avg loss: %f, avg accuracy: %f' % (EPOCH, total_loss, total_accuracy))
        print('epoch: %d, avg loss test: %f, avg accuracy test: %f' % (EPOCH, total_loss_test, total_accuracy_test))

    generated = [id2vocab[i] for i in sess.run(model.generate, feed_dict={model.X: [test_X[0]]})[0, 0, :]]
    ' '.join(generated)

    ' '.join([id2vocab[i] for i in test_Y[0]])
