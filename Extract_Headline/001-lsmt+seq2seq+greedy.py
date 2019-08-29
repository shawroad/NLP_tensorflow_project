"""

@file  : 001-lsmt+seq2seq+greedy.py

@author: xiaolu

@time  : 2019-08-15

"""
import json
from sklearn.utils import shuffle
import tensorflow as tf
import collections
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class Summarization:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size):
        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse)

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)  # 计算序列长度
        print(self.X_seq_len)

        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)  # 计算序列长度
        print(self.Y_seq_len)

        batch_size = tf.shape(self.X)[0]

        # 词嵌入
        encoder_embedding = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embedding = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embedding, self.X)

        # 编码
        encoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        self.encoder_out, self.encoder_state = tf.nn.dynamic_rnn(cell=encoder_cells,
                                                                 inputs=encoder_embedded,
                                                                 sequence_length=self.X_seq_len,
                                                                 dtype=tf.float32)

        encoder_state = tuple(self.encoder_state[-1] for _ in range(num_layers))  # 获取的是每层最后的隐态

        # 将self.Y中的样本一个一个提出来　并加上相应的标志
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)

        # 定义解码输出的那个部分
        dense = tf.layers.Dense(to_dict_size)  # 定义一个dense网络
        # 定义解码网络
        decoder_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])

        training_helper = tf.contrib.seq2seq.TrainingHelper(
            # 1. 输出进行词嵌入
            inputs=tf.nn.embedding_lookup(decoder_embedding, decoder_input),
            # 2. 获取序列的长度
            sequence_length=self.Y_seq_len,
            # 3. 主轴是否为时间
            time_major=False
        )
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=decoder_cells,
            helper=training_helper,
            initial_state=self.encoder_state,
            output_layer=dense
        )
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=training_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.Y_seq_len)
        )
        self.training_logits = training_decoder_output.rnn_output

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

        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=predicting_decoder,
            impute_finished=True,
            maximum_iterations=tf.reduce_max(self.X_seq_len)
        )

        self.predicting_ids = predicting_decoder_output.sample_id

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                     targets=self.Y,
                                                     weights=masks
                                                     )
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        y_t = tf.argmax(self.training_logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)

        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


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


# 主题模型
def topic_modelling(string, n = 500):
    vectorizer = TfidfVectorizer()
    tf = vectorizer.fit_transform([string])
    tf_features = vectorizer.get_feature_names()
    compose = TruncatedSVD(1).fit(tf)
    return ' '.join([tf_features[i] for i in compose.components_[0].argsort()[: -n - 1: -1]])


def bulid_dataset(words, n_words):
    '''

    :param words: 所有语料连起来　形成一个很大的字符串
    :param n_words: 不重复的词的个数　就词表的大小
    :return: data:, count, vocab2id, id2vocab
    '''
    # 预处理数据集
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words))
    vocab2id = dict()

    # 词与id 的映射
    for word, _ in count:
        vocab2id[word] = len(vocab2id)

    # 统计pad的长度
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


# 将文本转为id序列
def str_idx(corpus, dic, UNK=3):
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


if __name__ == '__main__':
    # 1. 加载数据
    with open('./dataset/ctexts.json', 'r') as fopen:
        ctexts = json.load(fopen)

    with open('./dataset/headlines.json', 'r') as fopen:
        headlines = json.load(fopen)

    h, c = [], []
    for i in range(len(ctexts)):
        try:
            c.append(topic_modelling(ctexts[i]))
            h.append(headlines[i])
        except:
            pass

    # 对文章进行处理
    concat_from = ' '.join(c).split()  # 将所有语料连起来
    vocabulary_size_from = len(list(set(concat_from)))  # 词表的大小
    data_from, count_from, vocab2id_from, id2vocab_from = bulid_dataset(concat_from, vocabulary_size_from)
    print('vocab from size: %d' % vocabulary_size_from)
    print('Most common words', count_from[4:10])
    print('Sample data', data_from[:10], [id2vocab_from[i] for i in data_from[:10]])

    # 对标题进行处理
    concat_to = ' '.join(h).split()
    vocabulary_size_to = len(list(set(concat_to)))
    data_to, count_to, vocab2id_to, id2vocab_to = bulid_dataset(concat_to, vocabulary_size_to)
    print('vocab to size: %d' % vocabulary_size_to)
    print('Most common words', count_to[4:10])
    print('Sample data', data_to[:10], [id2vocab_to[i] for i in data_to[:10]])

    # 在每个标题后面添加结束标志
    for i in range(len(h)):
        h[i] = h[i] + 'EOS'
    print(h[0])

    GO = vocab2id_from['GO']
    PAD = vocab2id_from['PAD']
    EOS = vocab2id_from['EOS']
    UNK = vocab2id_from['UNK']

    # 分别将文本和标题转为对应的id
    X = str_idx(c, vocab2id_from)
    Y = str_idx(h, vocab2id_to)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    size_layer = 128
    num_layers = 2
    embedded_size = 128
    batch_size = 8
    epoch = 5

    tf.reset_default_graph()
    sess = tf.Session()
    model = Summarization(size_layer, num_layers, embedded_size, len(vocab2id_from), len(vocab2id_to))
    sess.run(tf.global_variables_initializer())

    # 开始训练
    for epoch in range(10):
        total_loss, total_accuracy, total_loss_test, total_accuracy_test = 0, 0, 0, 0
        x_train, y_train = shuffle(x_train, y_train)
        print(len(x_train))   # 3516
        print(len(y_train))   # 3516
        print("*"*100)
        x_test, y_test = shuffle(x_test, y_test)
        print(len(x_test))   # 879
        print(len(y_test))   # 879

        for k in range(0, len(x_train), batch_size):
            batch_x, _ = pad_sentence_batch(x_train[k: min(k + batch_size, len(x_train))], PAD)
            batch_y, _ = pad_sentence_batch(y_train[k: min(k + batch_size, len(x_train))], PAD)

            acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                    feed_dict={model.X: batch_x, model.Y: batch_y})

            total_loss += loss
            total_accuracy += loss
            print("当前epoch:{}, 当前batch:{}, 损失:{}, 准确率:{}, 这是训练集上的表现".format(
                epoch, k//batch_size, loss, acc
            ))

        for k in range(0, len(x_test), batch_size):
            batch_x, _ = pad_sentence_batch(x_test[k: min(k + batch_size, len(x_test))], PAD)
            batch_y, _ = pad_sentence_batch(y_test[k: min(k + batch_size, len(y_test))], PAD)
            acc, loss = sess.run([model.accuracy, model.cost],
                                 feed_dict={model.X: batch_x, model.Y: batch_y})

            total_loss_test += loss
            total_accuracy_test += acc
            print("当前epoch:{}, 当前batch:{}, 损失:{}, 准确率:{}, 这是测试集上的表现".format(
                epoch, k // batch_size, loss, acc
            ))

            # 对测试集进行预测
            print(sess.run(model.predicting_ids, feed_dict={model.X: batch_x}))

        total_loss /= (len(x_train) / batch_size)
        total_accuracy /= (len(x_train) / batch_size)
        total_loss_test /= (len(x_test) / batch_size)
        total_accuracy_test /= (len(x_test) / batch_size)

        print('epoch: %d, avg loss: %f, avg accuracy: %f' % (epoch, total_loss, total_accuracy))
        print('epoch: %d, avg loss test: %f, avg accuracy test: %f' % (epoch, total_loss_test, total_accuracy_test))
