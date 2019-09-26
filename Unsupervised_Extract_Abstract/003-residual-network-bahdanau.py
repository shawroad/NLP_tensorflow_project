"""

@file  : 003-residual-network-bahdanau.py

@author: xiaolu

@time  : 2019-08-29

"""
import tensorflow as tf
import re
import collections
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def simple_textcleaning(string):
    # 简单清洗语料
    string = re.sub('[^A-Za-z ]+', ' ', string)
    return re.sub(r'[ ]+', ' ', string.lower()).strip()


def batch_sequence(sentences, dictionary, maxlen = 50):
    # 制作batch数据 并进行padding
    np_array = np.zeros((len(sentences), maxlen), dtype = np.int32)
    for no_sentence, sentence in enumerate(sentences):
        current_no = 0
        for no, word in enumerate(sentence.split()[: maxlen - 2]):
            np_array[no_sentence, no] = dictionary.get(word, 1)
            current_no = no
        np_array[no_sentence, current_no + 1] = 3
    return np_array


def counter_words(sentences):
    # 统计词频
    word_counter = collections.Counter()
    word_list = []
    num_lines, num_words = (0, 0)
    for i in sentences:
        words = re.findall('[\\w\']+|[;:\-\(\)&.,!?"]', i)
        word_counter.update(words)
        word_list.extend(words)
        num_lines += 1
        num_words += len(words)
    return word_counter, word_list, num_lines, num_words


def build_dict(word_counter, vocab_size = 50000):
    # 建立词典
    count = [['PAD', 0], ['UNK', 1], ['START', 2], ['END', 3]]
    count.extend(word_counter.most_common(vocab_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    return dictionary, {word: idx for idx, word in dictionary.items()}


def split_by_dot(string):
    # 按照标点进行分句
    string = re.sub(
        r'(?<!\d)\.(?!\d)',
        'SPLITTT',
        string.replace('\n', '').replace('/', ' '),
    )
    string = string.split('SPLITTT')
    return [re.sub(r'[ ]+', ' ', sentence).strip() for sentence in string]


class Attention:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.dense_layer = tf.layers.Dense(hidden_size)
        self.v = tf.random_normal([hidden_size], mean=0, stddev=1/np.sqrt(hidden_size))

    def score(self, hidden_tensor, encoder_outputs):

        energy = tf.nn.tanh(self.dense_layer(tf.concat([hidden_tensor, encoder_outputs], 2)))
        energy = tf.transpose(energy, [0, 2, 1])
        batch_size = tf.shape(encoder_outputs)[0]
        v = tf.expand_dims(tf.tile(tf.expand_dims(self.v, 0), [batch_size, 1]), 1)
        energy = tf.matmul(v, energy)

        return tf.squeeze(energy, 1)

    def __call__(self, hidden, encoder_outputs):
        seq_len = tf.shape(encoder_outputs)[1]
        H = tf.tile(tf.expand_dims(hidden, 1), [1, seq_len, 1])
        attn_energies = self.score(H, encoder_outputs)
        return tf.expand_dims(tf.nn.softmax(attn_energies), 1)


class Model:
    def __init__(self, dict_size, size_layers, learning_rate, maxlen, num_blocks=3):
        '''
        :param dict_size: 词表大小
        :param size_layers: 每步的输出维度
        :param learning_rate: 学习率
        :param maxlen: padding的长度
        :param num_blocks:
        '''

        block_size = size_layers

        # 1. 定义输入
        self.BEFORE = tf.placeholder(tf.int32, [None, maxlen])
        self.INPUT = tf.placeholder(tf.int32, [None, maxlen])
        self.AFTER = tf.placeholder(tf.int32, [None, maxlen])

        self.batch_size = tf.shape(self.INPUT)[0]

        self.output_layer = tf.layers.Dense(dict_size, name="output_layer")
        self.output_layer.build(size_layers)

        # 2. 对输入进行embedding
        self.embeddings = tf.Variable(tf.random_uniform([dict_size, size_layers], -1, 1))
        embedded = tf.nn.embedding_lookup(self.embeddings, self.INPUT)

        # 3. 引入注意力
        self.attention = Attention(size_layers)

        def residual_block(x, size, rate, block, reuse=False):

            with tf.variable_scope('block_%d_%d' % (block, rate), reuse=reuse):

                attn_weights = self.attention(tf.reduce_sum(x, axis=1), x)
                conv_filter = tf.layers.conv1d(
                    attn_weights,
                    x.shape[2] // 4,
                    kernel_size=size,
                    strides=1,
                    padding='same',
                    dilation_rate=rate,
                    activation=tf.nn.tanh,
                )
                conv_gate = tf.layers.conv1d(
                    x,
                    x.shape[2] // 4,
                    kernel_size=size,
                    strides=1,
                    padding='same',
                    dilation_rate=rate,
                    activation=tf.nn.sigmoid,
                )
                out = tf.multiply(conv_filter, conv_gate)
                out = tf.layers.conv1d(
                    out,
                    block_size,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    activation=tf.nn.tanh,
                )
                return tf.add(x, out), out

        forward = tf.layers.conv1d(embedded, block_size, kernel_size=1, strides=1, padding='SAME')

        zeros = tf.zeros_like(forward)
        for i in range(num_blocks):
            for r in [1, 2, 4, 8, 16]:
                forward, s = residual_block(
                    forward, size=7, rate=r, block=i
                )
                zeros = tf.add(zeros, s)
        forward = tf.layers.conv1d(
            zeros,
            block_size,
            kernel_size=1,
            strides=1,
            padding='SAME',
            activation=tf.nn.tanh,
        )
        self.get_thought = tf.reduce_sum(forward, axis=1, name='logits')

        def decoder(labels, reuse):
            decoder_in = tf.nn.embedding_lookup(self.embeddings, labels)
            forward = tf.layers.conv1d(decoder_in, block_size, kernel_size=1, strides=1, padding='SAME')

            zeros = tf.zeros_like(forward)

            for r in [8, 16, 24]:
                forward, s = residual_block(forward, size=7, rate=r, block=10, reuse=reuse)
                zeros = tf.add(zeros, s)

            return tf.layers.conv1d(zeros, block_size, kernel_size=1, strides=1, padding='SAME', activation=tf.nn.tanh)

        fw_logits = decoder(self.AFTER, False)
        bw_logits = decoder(self.BEFORE, True)

        self.attention = tf.matmul(
            self.get_thought, tf.transpose(self.embeddings), name='attention'
        )
        self.loss = self.calculate_loss(fw_logits, self.AFTER) + self.calculate_loss(bw_logits, self.BEFORE)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def calculate_loss(self, outputs, labels):
        # 计算损失
        mask = tf.cast(tf.sign(labels), tf.float32)
        logits = self.output_layer(outputs)
        return tf.contrib.seq2seq.sequence_loss(logits, labels, mask)


if __name__ == '__main__':
    # 1. 加载语料
    contents = []
    with open('books/Blood_Born') as fopen:
        contents.extend(split_by_dot(fopen.read()))

    with open('books/Dark_Thirst') as fopen:
        contents.extend(split_by_dot(fopen.read()))

    # 2. 进行清洗
    contents = [simple_textcleaning(sentence) for sentence in contents]
    contents = [sentence for sentence in contents if len(sentence) > 20]

    # 3. 定义一些超参数
    maxlen = 50
    vocabulary_size = len(set(' '.join(contents).split()))
    embedding_size = 256
    learning_rate = 1e-3
    batch_size = 16

    # 4. 讲语料句子分为　左　中　右
    stride = 1
    t_range = int((len(contents) - 3) / stride + 1)
    left, middle, right = [], [], []
    for i in range(t_range):
        slices = contents[i * stride: i * stride + 3]
        left.append(slices[0])
        middle.append(slices[1])
        right.append(slices[2])

    left, middle, right = shuffle(left, middle, right)   # 打乱语料

    # 5. 统计词频　并建立词表
    word_counter, _, _, _ = counter_words(middle)
    dictionary, _ = build_dict(word_counter, vocab_size=vocabulary_size)

    tf.reset_default_graph()
    sess = tf.Session()
    model = Model(len(dictionary), embedding_size, learning_rate, maxlen)
    sess.run(tf.global_variables_initializer())

    for i in range(5):
        for p in range(0, len(middle), batch_size):
            index = min(p + batch_size, len(middle))
            batch_x = batch_sequence(middle[p: index], dictionary, maxlen=maxlen)
            batch_y_before = batch_sequence(left[p: index], dictionary, maxlen=maxlen)
            batch_y_after = batch_sequence(right[p: index], dictionary, maxlen=maxlen)

            loss, _ = sess.run([model.loss, model.optimizer],
                               feed_dict={model.BEFORE: batch_y_before,
                                          model.INPUT: batch_x,
                                          model.AFTER: batch_y_after})
            print("epoch:{}, 第{}batch_size, 损失:{}".format(i, p // batch_size, loss))

    with open('books/Driftas_Quest') as f:
        book = f.read()

    book = split_by_dot(book)
    book = [simple_textcleaning(sentence) for sentence in book]
    book = [sentence for sentence in book if len(sentence) > 20][100:200]
    book_sequences = batch_sequence(book, dictionary, maxlen = maxlen)
    encoded, attention = sess.run([model.get_thought, model.attention],feed_dict={model.INPUT:book_sequences})

    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans = kmeans.fit(encoded)
    avg = []
    closest = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, encoded)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    print('. '.join([book[closest[idx]] for idx in ordering]))

    indices = np.argsort(attention.mean(axis=0))[::-1]
    rev_dictionary = {v: k for k, v in dictionary.items()}

    print([rev_dictionary[i] for i in indices[:10]])
