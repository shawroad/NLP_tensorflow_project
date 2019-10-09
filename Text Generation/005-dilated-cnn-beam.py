"""

@file  : 005-dilated-cnn-beam.py

@author: xiaolu

@time  : 2019-10-08

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensor2tensor.utils import beam_search
sns.set()


def start_sent(x):
    '''
    加起始的标志
    :param x:
    :return:
    '''
    _x = tf.fill([tf.shape(x)[0], 1], char2idx['<start>'])
    return tf.concat([_x, x], 1)


def end_sent(x):
    '''
    加结束的标志
    :param x:
    :return:
    '''
    _x = tf.fill([tf.shape(x)[0], 1], char2idx['<end>'])
    return tf.concat([x, _x], 1)


def embed_seq(x, vocab_sz, embed_dim, name, zero_pad=True):
    embedding = tf.get_variable(name, [vocab_sz, embed_dim])
    if zero_pad:
        embedding = tf.concat([tf.zeros([1, embed_dim]), embedding[1:, :]], 0)
    x = tf.nn.embedding_lookup(embedding, x)
    return x


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


def cnn_block(x, dilation_rate, pad_sz, hidden_dim, kernel_size):
    x = layer_norm(x)
    pad = tf.zeros([tf.shape(x)[0], pad_sz, hidden_dim])
    x = tf.layers.conv1d(inputs=tf.concat([pad, x, pad], 1),
                         filters=hidden_dim,
                         kernel_size=kernel_size,
                         dilation_rate=dilation_rate)
    x = x[:, :-pad_sz, :]
    x = tf.nn.relu(x)
    return x


# 定义模型
class Generator:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, kernel_size=5):
        '''
        :param size_layer:
        :param num_layers:
        :param embedded_size:
        :param dict_size:
        :param learning_rate:
        :param kernel_size:
        '''
        # 定义输入
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.training = tf.placeholder(tf.bool, None)
        self.dict_size = dict_size
        self.embedded_size = embedded_size
        self.size_layer = size_layer
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        batch_size = tf.shape(self.X)[0]
        x = start_sent(self.X)
        y = end_sent(self.Y)
        self.y = y

        logits = self.forward(x)
        self.logits = logits

        # 定义损失优化器
        self.cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=y,
            weights=tf.to_float(tf.ones_like(y))
        ))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)

        y_t = tf.argmax(logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)

        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def forward(self, x):
        # 词嵌入
        with tf.variable_scope('embed', reuse=tf.AUTO_REUSE):
            x = embed_seq(x, self.dict_size, self.embedded_size, 'word')
        x += position_encoding(x)

        for i in range(self.num_layers):
            dilation_rate = 2 ** i
            pad_sz = (self.kernel_size - 1) * dilation_rate

            with tf.variable_scope('block_%d' % i, reuse=tf.AUTO_REUSE):
                x += cnn_block(x, dilation_rate, pad_sz, self.size_layer, self.kernel_size)

        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            return tf.layers.dense(x, self.dict_size)


def beam_search_decoding(length=1000):
    '''
    :param length:
    :return:
    '''
    # 加起始标志
    initial_ids = tf.constant(char2idx['<start>'], tf.int32, [1])

    def symbols_to_logits(ids):
        logits = model.forward(ids)
        return logits[:, tf.shape(ids)[1] - 1, :]

    final_ids, final_probs, _ = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        5,
        length,
        len(char2idx),
        0.0,
        eos_id=char2idx['<end>']
    )

    return final_ids[0, 0, :]


if __name__ == '__main__':
    # 加载数据
    with open('./data/shakespeare.txt') as fopen:
        shakespeare = fopen.read()

    # 建立词表　　词:id
    char2idx = {c: i+3 for i, c in enumerate(set(shakespeare))}
    char2idx['<pad>'] = 0
    char2idx['<start>'] = 1
    char2idx['<end>'] = 2

    # id: 词
    idx2char = {v: k for k, v in char2idx.items()}

    # 超参数
    batch_size = 32
    sequence_length = 1000  # 知道1000个去预测
    step = 25

    X = [char2idx[char] for char in list(shakespeare)]

    len_win = sequence_length
    sequences = []
    for i in range(0, len(X) - len_win, step):
        clip = X[i: i+len_win]
        sequences.append(clip)
    sequences = np.array(sequences)
    print(sequences.shape)   # (44576, 1000)

    # 定义超参数
    learning_rate = 0.001
    epoch = 10
    num_layers = 4
    size_layer = 128
    possible_batch_id = range(len(X) - sequence_length - 1)

    tf.reset_default_graph()
    sess = tf.Session()
    model = Generator(size_layer, num_layers, size_layer, len(char2idx), learning_rate)
    model.generate = beam_search_decoding()

    sess.run(tf.global_variables_initializer())

    batch_x = sequences[:10]
    sess.run([model.accuracy, model.cost], feed_dict={model.X: batch_x, model.Y: batch_x, model.training: True})

    LOST, ACCURACY = [], []
    for e in range(epoch):
        total_cost, total_accuracy = 0, 0
        for i in range(0, len(sequences), batch_size):
            batch_x = sequences[i: min(i + batch_size, len(sequences))]
            _, accuracy, cost = sess.run([model.optimizer, model.accuracy, model.cost],
                                         feed_dict={
                                             model.X: batch_x,
                                             model.Y: batch_x,
                                             model.training: True
                                         })
            total_cost += cost
            total_accuracy += accuracy
            LOST.append(cost)
            ACCURACY.append(accuracy)
            print("epoch: %d, step: %d, loss: %f, accuracy: %f" % (e, i // batch_size, cost, accuracy))

        total_cost /= (len(sequences) / batch_size)
        total_accuracy /= (len(sequences) / batch_size)
        print('epoch %d, average cost %f, average accuracy %f'%(e + 1, total_cost, total_accuracy))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    EPOCH = np.arange(len(LOST))
    plt.plot(EPOCH, LOST)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1, 2, 2)
    plt.plot(EPOCH, ACCURACY)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    # 生成文本
    print(''.join([idx2char[i] for i in sess.run(model.generate)]))