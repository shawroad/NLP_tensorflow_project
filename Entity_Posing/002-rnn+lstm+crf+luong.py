"""

@file  : 002-rnn+lstm+crf+luong.py

@author: xiaolu

@time  : 2019-09-06

"""
import re
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


class Model:
    def __init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers):
        def cells(size, reuse=False):
            return tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse),
                output_keep_prob=dropout
            )

        def luong(embedded, size):
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=hidden_size_word, memory=embedded
            )
            return tf.contrib.seq2seq.AttentionWrapper(
                cell=cells(hidden_size_word),
                attention_mechanism=attention_mechanism,
                attention_layer_size=hidden_size_word,
            )

        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        self.labels = tf.placeholder(tf.int32, shape=[None, None])
        self.maxlen = tf.shape(self.word_ids)[1]
        self.lengths = tf.count_nonzero(self.word_ids, 1)

        self.word_embeddings = tf.Variable(tf.truncated_normal([len(word2idx), dim_word], stddev=1.0 / np.sqrt(dim_word)))
        self.char_embeddings = tf.Variable(tf.truncated_normal([len(char2idx), dim_char], stddev=1.0 / np.sqrt(dim_char)))

        word_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
        char_embedded = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)

        s = tf.shape(char_embedded)
        char_embedded = tf.reshape(char_embedded, shape=[s[0] * s[1], s[-2], dim_char])

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells(hidden_size_char),
                cell_bw=cells(hidden_size_char),
                inputs=char_embedded,
                dtype=tf.float32,
                scope='bidirectional_rnn_char_%d' % n
            )
            char_embedded = tf.concat((out_fw, out_bw), 2)

        output = tf.reshape(char_embedded[:, -1], shape=[s[0], s[1], 2 * hidden_size_char])

        word_embedded = tf.concat([word_embedded, output], axis=-1)

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw,state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=luong(word_embedded, hidden_size_word),
                cell_bw=luong(word_embedded, hidden_size_word),
                inputs=word_embedded,
                dtype=tf.float32,
                scope='bidirectional_rnn_word_%d' % n,
            )
            word_embedded = tf.concat((out_fw, out_bw), 2)

        logits = tf.layers.dense(word_embedded, len(idx2tag))
        y_t = self.labels
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, y_t, self.lengths)

        self.cost = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        mask = tf.sequence_mask(self.lengths, maxlen=self.maxlen)
        self.tags_seq, tags_score = tf.contrib.crf.crf_decode(
            logits, transition_params, self.lengths
        )
        self.tags_seq = tf.identity(self.tags_seq, name='logits')

        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(self.tags_seq, mask)
        mask_label = tf.boolean_mask(y_t, mask)
        correct_pred = tf.equal(self.prediction, mask_label)
        correct_index = tf.cast(correct_pred, tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def parse(file):
    '''
    加载文件并且解析
    :param file: 文件名
    :return: 词<->词性
    '''
    with open(file) as fopen:
        texts = fopen.read().split('\n')

    left, right = [], []
    for text in texts:
        if '-DOCSTART' in text or not len(text):
            continue
        splitted = text.split()
        left.append(splitted[0])
        right.append(splitted[-1])
    return left, right


def process_string(string):
    '''
    :param string:
    :return:
    '''
    string= re.sub('[^A-Za-z0-9\-\/ ]+', ' ', string).split()
    return ' '.join([to_title(y.strip()) for y in string])


def to_title(string):
    if string.isupper():
        string = string.title()
    return string


def parse_XY(texts, labels):
    '''
    整理词性表　　词表　　字符表　　并将文本转为对应的数字序列
    :param texts: 文本　词的一个列表
    :param labels: 词性的一个列表
    :return: 词转为id的序列　词性转为id的序列
    '''
    global word2idx, tag2idx, char2idx, word_idx, tag_idx, char_idx
    X, Y = [], []
    for no, text in enumerate(texts):
        text = text.lower()  # 当前这个单词转小写
        tag = labels[no]  # 取出对应的词性
        for c in text:   # 字符表
            if c not in char2idx:
                char2idx[c] = char_idx
                char_idx += 1
        if tag not in tag2idx:   # 词性表
            tag2idx[tag] = tag_idx
            tag_idx += 1
        Y.append(tag2idx[tag])   # 当前这个词的词性转为id的值
        if text not in word2idx:   # 词表
            word2idx[text] = word_idx
            word_idx += 1
        X.append(word2idx[text])  # 将词转为id的标号
    return X, np.array(Y)


def iter_seq(x):
    return np.array([x[i: i+seq_len] for i in range(0, len(x)-seq_len, 1)])


def to_train_seq(*args):
    '''
    :param args:  词转为的id的序列　　　词性转为id的序列
    :return:
    '''
    return [iter_seq(x) for x in args]


def generate_char_seq(batch):
    '''
    传进来是50一个块　总共有多少块
    然后将每块的单词转为字符序列
    :param batch:
    :return:
    '''
    x = [[len(idx2word[i]) for i in k] for k in batch]  # 得出每个单词的长度
    maxlen = max([j for i in x for j in i])  # 最大长度
    temp = np.zeros((batch.shape[0], batch.shape[1], maxlen), dtype=np.int32)

    for i in range(batch.shape[0]):
        for k in range(batch.shape[1]):
            for no, c in enumerate(idx2word[batch[i, k]]):
                temp[i, k, -1-no] = char2idx[c]
    return temp   # [文章数, 单词个数, maxlen(每个单词按字符转的id)]


def pred2label(pred):
    # 将预测结果转为标签
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p])
        out.append(out_i)
    return out


if __name__ == '__main__':
    left_train, right_train = parse('./data/eng.train')
    left_test, right_test = parse('./data/eng.testa')
    # print(left_train[:10])
    # print(right_train[:10])

    word2idx = {'PAD': 0, 'NUM': 1, 'UNK': 2}  # 词表
    tag2idx = {'PAD': 0}   # 词性表
    char2idx = {'PAD': 0}
    word_idx = 3
    tag_idx = 1
    char_idx = 1

    train_X, train_Y = parse_XY(left_train, right_train)
    test_X, test_Y = parse_XY(left_test, right_test)
    # print(train_X[:20])
    # print(train_Y[:20])

    idx2word = {idx: tag for tag, idx in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}

    seq_len = 50

    X_seq, Y_seq = to_train_seq(train_X, train_Y)   # 长度为50为一个段落
    X_char_seq = generate_char_seq(X_seq)
    print(X_seq.shape)  # (203571, 50)
    print(X_char_seq.shape)  # (203571, 50, 61)

    X_seq_test, Y_seq_test = to_train_seq(test_X, test_Y)
    X_char_seq_test = generate_char_seq(X_seq_test)
    print(X_seq_test.shape)   # (51312, 50)
    print(X_char_seq_test.shape)  # (51312, 50, 27)

    train_X, train_Y, train_char = X_seq, Y_seq, X_char_seq
    test_X, test_Y, test_char = X_seq_test, Y_seq_test, X_char_seq_test

    tf.reset_default_graph()
    sess = tf.Session()

    dim_word = 64
    dim_char = 128
    dropout = 0.8
    learning_rate = 1e-3
    hidden_size_char = 128
    hidden_size_word = 128
    num_layers = 2
    batch_size = 128

    model = Model(dim_word, dim_char, dropout, learning_rate,
                  hidden_size_char, hidden_size_word, num_layers)
    sess.run(tf.global_variables_initializer())

    for e in range(3):
        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        for i in range(0, len(train_X), batch_size):

            batch_x = train_X[i: min(i + batch_size, train_X.shape[0])]
            batch_char = train_char[i: min(i + batch_size, train_X.shape[0])]
            batch_y = train_Y[i: min(i + batch_size, train_X.shape[0])]

            acc, cost, _ = sess.run(
                [model.accuracy, model.cost, model.optimizer],
                feed_dict={
                    model.word_ids: batch_x,
                    model.char_ids: batch_char,
                    model.labels: batch_y
                },
            )
            train_loss += cost
            train_acc += acc
            print('train_data: epoch:{}, step:{}, loss:{}, accuracy:{}'.format(e, i//batch_size+1, cost, acc))

        for i in range(0, len(test_X), batch_size):
            batch_x = test_X[i: min(i + batch_size, test_X.shape[0])]
            batch_char = test_char[i: min(i + batch_size, test_X.shape[0])]
            batch_y = test_Y[i: min(i + batch_size, test_X.shape[0])]
            acc, cost = sess.run(
                [model.accuracy, model.cost],
                feed_dict={
                    model.word_ids: batch_x,
                    model.char_ids: batch_char,
                    model.labels: batch_y
                },
            )
            test_loss += cost
            test_acc += acc
            print('test_data: epoch:{}, step:{}, loss:{}, accuracy:{}'.format(e, i//batch_size+1, cost, acc))

        train_loss /= len(train_X) / batch_size
        train_acc /= len(train_X) / batch_size
        test_loss /= len(test_X) / batch_size
        test_acc /= len(test_X) / batch_size

        print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'
              % (e, train_loss, train_acc, test_loss, test_acc))

    real_Y, predict_Y = [], []
    for i in range(0, len(test_X), batch_size):
        batch_x = test_X[i: min(i + batch_size, test_X.shape[0])]
        batch_char = test_char[i: min(i + batch_size, test_X.shape[0])]
        batch_y = test_Y[i: min(i + batch_size, test_X.shape[0])]
        predicted = pred2label(
            sess.run(model.tags_seq,
                     feed_dict={
                         model.word_ids: batch_x,
                         model.char_ids: batch_char,
                     },
                     )
        )
        real = pred2label(batch_y)
        predict_Y.extend(predicted)
        real_Y.extend(real)

    print(classification_report(np.array(real_Y).ravel(), np.array(predict_Y).ravel()))