"""

@file  : 001-skip-thought.py

@author: xiaolu

@time  : 2019-08-28

"""
import tensorflow as tf
import re
import collections
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class Model:
    def __init__(self, maxlen=50, vocabulary_size=20000, learning_rate=1e-3, embedding_size=256):
        '''
        :param maxlen: 句子长度
        :param vocabulary_size: 词表大小
        :param learning_rate: 学习率
        :param embedding_size: 词嵌入的维度
        '''
        self.output_size = embedding_size
        self.maxlen = maxlen

        # 准备词嵌入
        word_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -np.sqrt(3), np.sqrt(3)))

        # 记录训练步
        self.global_step = tf.get_variable('global_step', shape=[], trainable=False, initializer=tf.initializers.zeros())

        self.embeddings = word_embeddings
        self.output_layer = tf.layers.Dense(vocabulary_size, name='output_layer')

        # 前 中　后
        self.BEFORE = tf.placeholder(tf.int32, [None, maxlen])
        self.INPUT = tf.placeholder(tf.int32, [None, maxlen])
        self.AFTER = tf.placeholder(tf.int32, [None, maxlen])

        self.batch_size = tf.shape(self.INPUT)[0]

        self.get_thought = self.thought(self.INPUT)  # 将input通过一个双向的GRU

        self.attention = tf.matmul(
            self.get_thought, tf.transpose(self.embeddings), name='attention'
        )   # 将输出和词嵌入进行相似度量

        fw_logits = self.decoder(self.get_thought, self.AFTER)
        bw_logits = self.decoder(self.get_thought, self.BEFORE)
        self.loss = self.calculate_loss(fw_logits, self.AFTER) + self.calculate_loss(bw_logits, self.BEFORE)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def get_embedding(self, inputs):
        '''
        进行词嵌入
        :param inputs: 输入
        :return:
        '''
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def thought(self, inputs):
        encoder_in = self.get_embedding(inputs)
        fw_cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        bw_cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        sequence_length = tf.reduce_sum(tf.sign(inputs), axis=1)
        rnn_output = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, encoder_in, sequence_length=sequence_length,
            dtype=tf.float32
        )[1]
        return sum(rnn_output)

    def decoder(self, thought, labels):
        main = tf.strided_slice(labels, [0, 0], [self.batch_size, -1], [1, 1])
        shifted_labels = tf.concat([tf.fill([self.batch_size, 1], 2), main], 1)  # 每个序列填充开始的标志

        decoder_in = self.get_embedding(shifted_labels)  # 词嵌入
        cell = tf.nn.rnn_cell.GRUCell(self.output_size)
        max_seq_lengths = tf.fill([self.batch_size], self.maxlen)

        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_in, max_seq_lengths, time_major=False
        )

        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, thought)
        decoder_out = tf.contrib.seq2seq.dynamic_decode(decoder)[0].rnn_output

        return decoder_out

    def calculate_loss(self, outputs, labels):
        '''
        计算损失
        :param outputs:
        :param labels:
        :return:
        '''
        mask = tf.cast(tf.sign(labels), tf.float32)
        logits = self.output_layer(outputs)
        return tf.contrib.seq2seq.sequence_loss(logits, labels, mask)


def simple_textcleaning(string):
    string = re.sub('[^A-Za-z ]+', ' ', string)
    return re.sub(r'[ ]+', ' ', string.lower()).strip()


def batch_sequence(sentences, dictionary, maxlen=50):
    '''
    对序列进行填充
    :param sentence:
    :param dictionary:
    :param maxlen:
    :return:
    '''
    np_array = np.zeros((len(sentences), maxlen), dtype=np.int32)
    for no_sentence, sentence in enumerate(sentences):
        current_no = 0
        for no, word in enumerate(sentence.split()[: maxlen-2]):  # 减2是为了填充开始与结束
            np_array[no_sentence, no] = dictionary.get(word, 1)
            current_no = no
        np_array[no_sentence, current_no + 1] = 3   # 结束
    return np_array


def counter_words(sentences):
    '''
    统计词的
    :param sentences:
    :return:
    '''
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


def build_dict(word_counter, vocab_size=50000):
    '''
    建立词表
    :param word_counter:
    :param vocab_size:
    :return:
    '''
    count = [['PAD', 0], ['UNK', 1], ['START', 2], ['END', 3]]
    count.extend(word_counter.most_common(vocab_size))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    return dictionary, {word: idx for idx, word in dictionary.items()}


def split_by_dot(string):
    '''
    将传入预语料的标点去掉　并通过标点分割　得到每句话
    :param string:
    :return:
    '''
    string = re.sub(r'(?<!\d)\.(?!\d)', 'SPLITTT', string.replace('\n', '').replace('/', ' '))
    string = string.split('SPLITTT')
    return [re.sub(r'[ ]+', ' ', sentence).strip() for sentence in string]


if __name__ == '__main__':
    contents = []
    with open('./books/Blood_Born') as fopen:
        contents.extend(split_by_dot(fopen.read()))

    with open('./books/Dark_Thirst') as fopen:
        contents.extend(split_by_dot(fopen.read()))

    maxlen = 50
    vocabulary_size = len(set(' '.join(contents).split()))  # 得到词表的大小
    embedding_size = 256
    learning_rate = 1e-3
    batch_size = 16
    print("词表的大小:", vocabulary_size)

    stride = 1
    t_range = int((len(contents) - 3) / stride + 1)

    left, middle, right = [], [], []

    for i in range(t_range):
        slices = contents[i * stride: i * stride + 3]
        left.append(slices[0])
        middle.append(slices[1])
        right.append(slices[2])

    # 每次截断三句话　　分　左, 中, 右
    left, middle, right = shuffle(left, middle, right)

    word_counter, _, _, _ = counter_words(middle)
    print(word_counter)  # [{词: 个数, 词:　个数...}]

    dictionary, _ = build_dict(word_counter, vocab_size=vocabulary_size)

    tf.reset_default_graph()
    sess = tf.Session()

    model = Model(vocabulary_size=len(dictionary),
                  embedding_size=embedding_size)
    sess.run(tf.global_variables_initializer())

    for i in range(5):
        for p in range(0, len(middle), batch_size):
            index = min(p + batch_size, len(middle))
            batch_x = batch_sequence(middle[p: index], dictionary, maxlen=maxlen)
            batch_y_before = batch_sequence(left[p: index], dictionary, maxlen=maxlen)
            batch_y_after = batch_sequence(right[p: index], dictionary, maxlen=maxlen)

            loss, _ = sess.run([model.loss, model.optimizer], feed_dict={model.BEFORE: batch_y_before,
                                                                         model.INPUT: batch_x,
                                                                         model.AFTER: batch_y_after})
            print("损失:{}".format(loss))

    # 读取新语料进行测试
    with open('./books/Driftas_Quest') as f:
        book = f.read()

    book = split_by_dot(book)  # 分句
    book = [simple_textcleaning(sentence) for sentence in book]  # 简单清洗
    book = [sentence for sentence in book if len(sentence) > 20][100:200]  # 截断长度
    book_sequences = batch_sequence(book, dictionary, maxlen=maxlen)   # 填充
    encoded, attention = sess.run([model.get_thought, model.attention],
                                  feed_dict={model.INPUT: book_sequences})

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

    print("*"*100)

    indices = np.argsort(attention.mean(axis=0))[::-1]

    rev_dictionary = {v: k for k, v in dictionary.items()}

    print([rev_dictionary[i] for i in indices[:10]])
