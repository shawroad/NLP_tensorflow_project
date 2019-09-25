"""

@file  : 004-gru-seq2seq-greedy-word.py

@author: xiaolu

@time  : 2019-09-24

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import collections
sns.set()


# 定义模型
class Generator:
    def __init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning, batch_size):
        '''
        :param size_layer:
        :param num_layers:
        :param embedded_size:
        :param from_dict_size:
        :param to_dict_size:
        :param learning:
        :param batch_size:
        '''
        def cells(reuse=False):
            return tf.nn.rnn_cell.GRUCell(size_layer, reuse=reuse)

        # 1. 定义输入
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)
        batch_size = tf.shape(self.X)[0]

        # 2. embedding
        encoder_embedding = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embedding = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))

        # 3. 多层rnn
        self.cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        # 初始状态
        self.encoder_state = self.cells.zero_state(
            dtype=tf.float32, batch_size=tf.shape(self.X)[0]
        )

        # 编码
        _, encoder_state = tf.nn.dynamic_rnn(
            cell=self.cells,
            inputs=tf.nn.embedding_lookup(encoder_embedding, self.X),
            sequence_length=self.X_seq_len,
            initial_state=self.encoder_state,
            dtype=tf.float32
        )

        # 解码
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], 0), main], 1)   # 给解码的输入的前面进行填充　开始标志

        dense = tf.layers.Dense(to_dict_size)
        decoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(decoder_embedding, decoder_input),
            sequence_length=self.Y_seq_len,
            time_major=False
        )

        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells,
            helper=training_helper,
            initial_state=encoder_state,
            output_layer=dense
        )

        training_decoder_output, self.training_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.Y_seq_len)
        )

        self.training_logits = training_decoder_output.rnn_output

        # 推理
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=decoder_embedding,
            start_tokens=tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),
            end_token=EOS
        )

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells,
            helper=predicting_helper,
            initial_state=encoder_state,
            output_layer=dense
        )

        predicting_decoder_output, self.predict_state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.X_seq_len)
        )

        self.predicting_ids = predicting_decoder_output.sample_id

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        # 损失
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
    '''
    整理词表
    :param words:
    :param n_words:
    :param atleast:
    :return:
    '''
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)
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


def train_random_batch():
    '''
    训练
    :return:
    '''
    LOST, ACCURACY = [], []
    # 整理一批数据　　类似与padding
    batch_x = np.zeros((batch_size, sequence_length))
    for n in range(batch_size):
        index = np.random.randint(0, len(data) - sequence_length - 1)
        batch_x[n] = data[index: index + sequence_length]
    initial_state = sess.run(model.predict_state, feed_dict={model.X: batch_x})

    for i in range(epoch):
        batch_x = np.zeros((batch_size, sequence_length))
        batch_y = np.zeros((batch_size, sequence_length + 1))

        for n in range(batch_size):
            index = np.random.randint(0, len(data) - sequence_length - 1)
            batch_x[n] = data[index: index + sequence_length]
            batch_y[n] = data[index + 1: index + sequence_length + 1] + [EOS]

        accuracy, _, loss, initial_state = sess.run([model.accuracy, model.optimizer,
                                                     model.cost, model.predict_state],
                                                    feed_dict={
                                                        model.X: batch_x,
                                                        model.Y: batch_y,
                                                        model.encoder_state: initial_state
                                                    })
        ACCURACY.append(accuracy)
        LOST.append(loss)
        print("epoch: %d; loss: %f; accuracy: %f" % (i, loss, accuracy))
    return LOST, ACCURACY


def generate_based_sequence(length_sentence):
    '''
    生成文本
    :param length_sentence:
    :return:
    '''
    index = np.random.randint(0, len(data) - sequence_length - 1)
    x = np.array([data[index:index + sequence_length]])
    initial_state, ids = sess.run([model.predict_state, model.predicting_ids],
                                  feed_dict={model.X: x})
    ids = ids[0].tolist()

    while len(ids) < length_sentence:
        initial_state, ids_ = sess.run([model.predict_state, model.predicting_ids],
                                       feed_dict={model.X: [ids[-sequence_length:]],
                                                  model.encoder_state: initial_state})
        ids.extend(ids_[0].tolist())

    return ' '.join([id2vocab[i] for i in ids])


if __name__ == '__main__':
    # 1. 加载语料
    with open('./data/shakespeare.txt') as fopen:
        shakespeare = fopen.read().split()

    vocabulary_size = len(list(set(shakespeare)))
    data, count, vocab2id, id2vocab = build_dataset(shakespeare, vocabulary_size)

    GO = vocab2id['GO']
    PAD = vocab2id['PAD']
    EOS = vocab2id['EOS']
    UNK = vocab2id['UNK']

    # 定义一些超参数
    learning_rate = 0.001
    batch_size = 32
    sequence_length = 64
    epoch = 3000
    num_layers = 2
    size_layer = 256
    possible_batch_id = range(len(data) - sequence_length - 1)

    tf.reset_default_graph()
    sess = tf.Session()
    model = Generator(size_layer, num_layers, size_layer, len(vocab2id),
                      len(vocab2id), learning_rate, batch_size)

    sess.run(tf.global_variables_initializer())

    LOST, ACCURACY = train_random_batch()

    # 画出损失和准确率
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

    print(generate_based_sequence(1000))
