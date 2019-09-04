"""

@file  : 002-lstm-birnn-seq2seq-beam-luong.py

@author: xiaolu

@time  : 2019-08-29

"""
import json
import tensorflow as tf
import collections
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from sklearn.utils import shuffle
import time


class Summarization:
    def __init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size, grad_clip=5.0,
                 beam_width=5, force_teaching_ratio=0.5):
        '''
        :param size_layer: 每步输出的维度
        :param num_layers: 准备搞几层
        :param embedded_size: 词嵌入维度
        :param from_dict_size: 文本词表大小
        :param to_dict_size: 标题词表大小
        :param batch_size: 批量
        :param grad_clip:
        :param beam_width: beam_search的宽度
        :param force_teaching_ratio:
        '''
        def lstm_cell(size, reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse)

        # 1. 定义输入
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)

        batch_size = tf.shape(self.X)[0]

        # 2. 词嵌入
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))

        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        # 编码
        # 多层双向lstm
        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell(size_layer // 2),
                cell_bw=lstm_cell(size_layer // 2),
                inputs=encoder_embedded,
                sequence_length=self.X_seq_len,
                dtype=tf.float32,
                scope='bidirectional_rnn_%d' % n
            )
            encoder_embedded = tf.concat((out_fw, out_bw), 2)

        bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
        bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)

        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)  # 等会作为解码的初始状态
        encoder_state = tuple([bi_lstm_state] * num_layers)

        # 解码
        with tf.variable_scope('decode'):
            # 解码加入Luong注意力
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=size_layer,
                memory=encoder_embedded,
                memory_sequence_length=self.X_seq_len
            )
            # 解码用单向多层的LSTM解码单元
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=size_layer
            )

            # 在每个标题前面加入GO 表示开始
            main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)

            training_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=tf.nn.embedding_lookup(decoder_embeddings, decoder_input),
                sequence_length=self.Y_seq_len,
                embedding=decoder_embeddings,
                sampling_probability=1-force_teaching_ratio,  # 采样概率
                time_major=False
            )

            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=decoder_cell,
                helper=training_helper,
                initial_state=decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state),
                output_layer=tf.layers.Dense(to_dict_size)
            )

            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=tf.reduce_max(self.Y_seq_len)
            )

            self.logits = training_decoder_output.rnn_output  # 得到解码的输出

        with tf.variable_scope('decode', reuse=True):
            # 推理的过程吧
            encoder_out_tiled = tf.contrib.seq2seq.tile_batch(encoder_embedded, beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
            X_seq_len_tiled = tf.contrib.seq2seq.tile_batch(self.X_seq_len, beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=size_layer,
                memory=encoder_out_tiled,
                memory_sequence_length=X_seq_len_tiled
            )
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer, reuse=True) for _ in range(num_layers)]),
                attention_mechanism=attention_mechanism,
                attention_layer_size=size_layer)

            # 通过beam_search进行解码
            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,   # 解码单元
                embedding=decoder_embeddings,  # 词嵌入的维度
                start_tokens=tf.tile(tf.constant([GO], dtype=tf.int32), [batch_size]),  # 开始标志
                end_token=EOS,  # 结束标志
                initial_state=decoder_cell.zero_state(batch_size * beam_width, tf.float32).clone(
                    cell_state=encoder_state_tiled
                ),   # 初始状态
                beam_width=beam_width,   # beam_search的宽度
                output_layer=tf.layers.Dense(to_dict_size, _reuse=True),  # 输出的那个dense
                length_penalty_weight=0.0
            )

            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=False,
                maximum_iterations=tf.reduce_max(self.X_seq_len)
            )

            self.predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]

        # 计算seq2seq中的mask损失
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)

        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.Y, weights=masks)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cost)

        y_t = tf.argmax(self.logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)

        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)

        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def topic_modelling(string, n=500):
    vectorizer = TfidfVectorizer()
    tf = vectorizer.fit_transform([string])
    tf_features = vectorizer.get_feature_names()
    compose = TruncatedSVD(1).fit(tf)
    return ' '.join([tf_features[i] for i in compose.components_[0].argsort()[: -n - 1 : -1]])


def build_dataset(words, n_words):
    '''
    :param words: 所有词用空格连接起来的大字符串
    :param n_words:　去重后的词个数 
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
    id2vocab = dict(zip(vocab2id.values(), vocab2id.keys()))
    return data, count, vocab2id, id2vocab


def str_idx(corpus, dic, UNK=3):
    '''
    将文本转为数字序列
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
    '''
    进行padding
    :param sentence_batch:
    :param pad_int:
    :return:
    '''
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


if __name__ == '__main__':
    # 1. 加载数据    
    with open('./dataset/ctexts.json', 'r') as fopen:
        ctexts = json.load(fopen)
    
    with open('./dataset/headlines.json', 'r') as fopen:
        headlines = json.load(fopen)
    
    h, c = [], []
    for i in range(len(ctexts)):
        try:
            c.append(topic_modelling(ctexts[i]))  # 先对原始文本做tfidf+矩阵分解
            h.append(headlines[i])
        except:
            pass

    # 文本
    concat_from = ' '.join(c).split()  # 分词
    vocabulary_size_from = len(list(set(concat_from)))  # 去重后的词
    data_from, count_from, vocab2id_from, id2vocab_from = build_dataset(concat_from, vocabulary_size_from)
    print('vocab from size: %d' % vocabulary_size_from)
    print('Most common words', count_from[4:10])
    print('Sample data', data_from[:10], [id2vocab_from[i] for i in data_from[:10]])

    # 标题
    concat_to = ' '.join(h).split()
    vocabulary_size_to = len(list(set(concat_to)))
    data_to, count_to, vocab2id_to, id2vocab_to = build_dataset(concat_to, vocabulary_size_to)
    print('vocab to size: %d' % vocabulary_size_to)
    print('Most common words', count_to[4:10])
    print('Sample data', data_to[:10], [id2vocab_to[i] for i in data_to[:10]])

    for i in range(len(h)):
        h[i] = h[i] + ' EOS'  # 给每个标题最后加EOS

    GO = vocab2id_from['GO']
    PAD = vocab2id_from['PAD']
    EOS = vocab2id_from['EOS']
    UNK = vocab2id_from['UNK']

    # 将文本序列转为数字序列
    X = str_idx(c, vocab2id_from)
    Y = str_idx(h, vocab2id_to)

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.05)

    # 定义一些超参数
    size_layer = 128
    num_layers = 2
    embedded_size = 128
    batch_size = 8
    epoch = 20

    tf.reset_default_graph()
    sess = tf.Session()
    model = Summarization(size_layer, num_layers, embedded_size, len(vocab2id_from), len(vocab2id_to), batch_size)
    sess.run(tf.global_variables_initializer())

    for EPOCH in range(10):
        total_loss, total_accuracy, total_loss_test, total_accuracy_test = 0, 0, 0, 0

        train_X, train_Y = shuffle(train_X, train_Y)
        test_X, test_Y = shuffle(test_X, test_Y)

        for k in range(0, len(train_X), batch_size):
            batch_x, _ = pad_sentence_batch(train_X[k: min(k + batch_size, len(train_X))], PAD)
            batch_y, _ = pad_sentence_batch(train_Y[k: min(k + batch_size, len(train_X))], PAD)

            acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                    feed_dict={model.X: batch_x, model.Y: batch_y})

            total_loss += loss
            total_accuracy += acc
            print("当前epoch:{}, 当前batch:{}, 损失:{}, 准确率:{}, 这是训练集上的表现".format(
                EPOCH, k//batch_size, loss, acc
            ))

        for k in range(0, len(test_X), batch_size):
            batch_x, _ = pad_sentence_batch(test_X[k: min(k + batch_size, len(test_X))], PAD)
            batch_y, _ = pad_sentence_batch(test_Y[k: min(k + batch_size, len(test_X))], PAD)
            acc, loss = sess.run([model.accuracy, model.cost],
                                 feed_dict={model.X: batch_x,
                                            model.Y: batch_y})
            total_loss_test += loss
            total_accuracy_test += acc
            print("当前epoch:{}, 当前batch:{}, 损失:{}, 准确率:{}, 这是测试集上的表现".format(
                EPOCH, k // batch_size, loss, acc
            ))

            # 对测试集进行预测
            print(sess.run(model.predicting_ids, feed_dict={model.X: batch_x}))

        total_loss /= (len(train_X) / batch_size)
        total_accuracy /= (len(train_X) / batch_size)
        total_loss_test /= (len(test_X) / batch_size)
        total_accuracy_test /= (len(test_X) / batch_size)

        print('epoch: %d, avg loss: %f, avg accuracy: %f' % (EPOCH, total_loss, total_accuracy))
        print('epoch: %d, avg loss test: %f, avg accuracy test: %f' % (EPOCH, total_loss_test, total_accuracy_test))


