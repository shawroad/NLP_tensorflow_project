"""

@file  : 004-lstm-birnn-seq2seq-luong.py

@author: xiaolu

@time  : 2019-07-23

"""
import re
import collections
import json
import tensorflow as tf


def build_dataset(words, n_words, atleast=1):
    '''
    :param words: 所有的词
    :param n_words: 去重后的词
    :param atleast: 过滤一些低频次
    :return:
    '''

    # 建立词表
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]
    count.extend(counter)

    vocab2id = {}
    for word, _ in count:
        vocab2id[word] = len(vocab2id)

    # 将所有词转为对应的数字
    data = []
    for word in words:
        index = vocab2id.get(word, 0)
        data.append(index)

    id2vocab = dict(zip(vocab2id.values(), vocab2id.keys()))

    return data, count, vocab2id, id2vocab


def clean_text(text):
    # 文本清洗
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return ' '.join([i.strip() for i in filter(None, text.split())])


# define model
class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate):
        '''
        :param size_layer: 每步的输出
        :param num_layers: 有多少层
        :param embedded_size: 词嵌入的维度
        :param from_dict_size: 问题字典
        :param to_dict_size: 回答字典
        :param learning_rate: 学习率
        '''
        def cells(size, reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(),
                                           reuse=reuse)

        # 定义输入
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])

        batch_size = tf.shape(self.X)[0]

        # 对问题和回答进行词嵌入
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        main = tf.strided_slice(self.X, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)
        decoder_embedded = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        def attention():
            # 在编码的cell中添加luong注意力
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=size_layer // 2,
                                                                    memory=encoder_embedded)
            return tf.contrib.seq2seq.AttentionWrapper(cell=cells(size_layer // 2),
                                                       attention_mechanism=attention_mechanism,
                                                       attention_layer_size=size_layer // 2)

        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=attention(),
                cell_bw=attention(),
                inputs=encoder_embedded,
                sequence_length=self.X_seq_len,
                dtype=tf.float32,
                scope='bidirectional_rnn_%d' % (n))
            encoder_embedded = tf.concat((out_fw, out_bw), 2)

        # 将多层最后输出的c , h 拼接起来
        bi_state_c = tf.concat((state_fw[0].c, state_bw[0].c), -1)
        bi_state_h = tf.concat((state_fw[0].h, state_bw[0].h), -1)

        # 将其重复多次, 因为我们解码也是用的是多层lstm
        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
        last_state = tuple([bi_lstm_state] * num_layers)

        # 解码
        with tf.variable_scope("decoder"):
            rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer) for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells_dec, decoder_embedded,
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


def str_idx(corpus, dic):
    # 将corpus语料转为id序列
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k,UNK))
        X.append(ints)
    return X


def pad_sentence_batch(sentence_batch, pad_int, maxlen):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
    return padded_seqs, seq_lens


if __name__ == '__main__':
    # 1. 加载语料 并整理成  问题<=>回答
    # 标号 与 文本 的对应
    lines = open('./data/movie_lines.txt', encoding='utf8', errors='ignore').read().split('\n')

    # 一组对话的标号
    conv_lines = open('./data/movie_conversations.txt', encoding='utf8', errors='ignore').read().split('\n')

    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]   # id 以及 对应的 句子

    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))

    questions = []
    answers = []

    for conv in convs:
        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])

    # 2. 对文本进行简单清洗
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # 3. 因为电脑资源有限, 所以我们这里只训练句子长短在为2到5之间的语料
    min_line_length = 2
    max_line_length = 5
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for question in clean_questions:
        # 通过问题长短进行过滤
        if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
            short_questions_temp.append(question)
            short_answers_temp.append(clean_answers[i])
        i += 1

    short_questions = []
    short_answers = []
    i = 0
    for answer in short_answers_temp:
        if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
            short_answers.append(answer)
            short_questions.append(short_answers_temp[i])
        i += 1

    # 训练集
    short_questions = short_questions[:500]
    short_answers = short_answers[:500]

    # 测试集
    question_test = short_questions[500:550]
    answer_test = short_answers[500:550]

    # 4. 建立词典  针对的是问题
    concat_from = ' '.join(short_questions + question_test).split()
    vocabulary_size_from = len(list(set(concat_from)))   # 这里是去重后的词
    data, count, vocab2id, id2vocab = build_dataset(concat_from, vocabulary_size_from)

    print("词表的大小:", len(vocab2id))
    print("常用的词:", count[4: 10])  # 前四个是标记
    print("部分样本数据:", data[:10], [id2vocab[i] for i in data[:10]])

    #  建立词典  针对的是回答
    concat_to = ' '.join(short_answers + answer_test).split()
    vocabulary_size_to = len(list(set(concat_to)))
    data_r, count_r, vocab2id_r, id2vocab_r = build_dataset(concat_to, vocabulary_size_to)

    print("词表的大小:", len(vocab2id_r))
    print("常用的词:", count_r[4: 10])  # 前四个是标记
    print("部分样本数据:", data_r[:10], [id2vocab_r[i] for i in data_r[:10]])

    GO = vocab2id['GO']
    PAD = vocab2id['PAD']
    EOS = vocab2id['EOS']
    UNK = vocab2id['UNK']

    print(GO, PAD, EOS, UNK)

    # 给每个回答加结束标志
    for i in range(len(short_answers)):
        short_answers[i] += ' EOS'

    process_question = [data, count, vocab2id, id2vocab]
    json.dump(process_question, open('question.json', 'w'))

    process_answer = [data_r, count_r, vocab2id_r, id2vocab_r]
    json.dump(process_answer, open('answer.json', 'w'))

    # 测试集，验证集 中 将问题 和 回答 转为id序列
    X = str_idx(short_questions, vocab2id)
    Y = str_idx(short_answers, vocab2id_r)
    X_test = str_idx(question_test, vocab2id)
    Y_test = str_idx(answer_test, vocab2id_r)

    # 进行序列的padding

    size_layer = 256
    num_layers = 2
    embedded_size = 128
    learning_rate = 0.001
    batch_size = 16
    epoch = 20

    maxlen_question = max([len(x) for x in X]) * 2   # 每个问题的长度*2
    maxlen_answer = max([len(y) for y in Y]) * 2   # 每个回答的长度*2

    tf.reset_default_graph()
    sess = tf.Session()

    model = Chatbot(size_layer, num_layers, embedded_size, len(vocab2id),
                    len(vocab2id_r), learning_rate)

    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        total_loss, total_accuracy = 0, 0
        for k in range(0, len(short_questions), batch_size):
            index = min(k + batch_size, len(short_questions))
            batch_x, seq_x = pad_sentence_batch(X[k: index], PAD, maxlen_answer)
            batch_y, seq_y = pad_sentence_batch(Y[k: index], PAD, maxlen_answer)
            predicted, accuracy, loss, _ = sess.run([tf.argmax(model.logits, 2),
                                                     model.accuracy, model.cost, model.optimizer],
                                                    feed_dict={model.X: batch_x,
                                                               model.Y: batch_y,
                                                               model.X_seq_len: seq_x,
                                                               model.Y_seq_len: seq_y})
            total_loss += loss
            total_accuracy += accuracy

        total_loss /= (len(short_questions) / batch_size)
        total_accuracy /= (len(short_questions) / batch_size)
        print('epoch: %d, avg loss: %f, avg accuracy: %f' % (i + 1, total_loss, total_accuracy))
