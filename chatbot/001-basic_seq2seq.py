"""

@file   : 001-basic_seq2seq.py

@author : xiaolu

@time1  : 2019-05-24

"""
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import re
import time
import collections
import os


def build_dataset(words, n_words, atleast=1):
    # 建立词表 以及词表与id之间的映射
    count = [['PAD', 0], ['GO', 1], ['EOS', 2], ['UNK', 3]]
    counter = collections.Counter(words).most_common(n_words)
    counter = [i for i in counter if i[1] >= atleast]  # 词频低于多少次将其扔掉不要
    count.extend(counter)   # 统计词频然后搞词表
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0  # 出现不知道的字符
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count   # 统计了没有编号的字符
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


lines = open('./data/movie_lines.txt', encoding='utf8', errors='ignore').read().split('\n')
conv_lines = open('./data/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(_line.split(','))

questions = []
answers = []


for conv in convs:
    for i in range(len(conv) - 1):
        questions.append(id2line[conv[i]])   # 问题
        answers.append(id2line[conv[i+1]])   # 回答


# 查看一下数据  分别输出前三个问题及回答
for q, a in zip(questions[:3], answers[:3]):
    print("问题:", q)
    print("回答:", a)

# 对话总共有221616条
print(len(questions))   # 221616
print(len(answers))  # 221616


def clean_text(text):
    # 整理语料  将一些缩写改成正常的形式 去除特殊的字符
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


# 预处理问题语料
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# 预处理回答语料
clean_answers = []
for answer in answers:
    clean_answers.append(answer)


# 为了模型训练快速 我们只将问题和答案在2个词到5个词之间的句子挑出来训练
min_line_length = 2   # 问题中含词最小2个
max_line_length = 5   # 问题中含词最多5个

# 先通过问题的长短过滤对话
short_questions_temp = []
short_answers_temp = []
i = 0
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)   # 问题加进列表
        short_answers_temp.append(clean_answers[i])   # 将其回答加入列表
    i += 1


# 再通过问题的长短过滤对话
short_questions = []
short_answers = []
i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1


print(len(short_questions))   # 23886
print(len(short_answers))  # 23886
# 查看一下数据  分别输出前三个问题及回答
for q, a in zip(short_questions[:3], short_answers[:3]):
    print("问题:", q)
    print("回答:", a)


# 切分数据集  为了模型好训练 这里训练集只有500句话  测试集有50句话
question_test = short_questions[500:550]
answer_test = short_answers[500:550]
short_questions = short_questions[:500]
short_answers = short_answers[:500]

# 问题中词表的映射
concat_from = ' '.join(short_questions+questions).split()
vocabulary_size_from = len(list(set(concat_from)))  # 词表的大小
data_from, count_from, dictionary_from, rev_dictionary_from = build_dataset(concat_from, vocabulary_size_from)
print('词的总数量:', vocabulary_size_from)
print('出现频率最高的6个词:', count_from[4:10])
print('id和词之间的对应:', data_from[:10], [rev_dictionary_from[i] for i in data_from[:10]])
print('词表的大小', len(dictionary_from))
print("词表中词的利用比:{}".format(round(len(dictionary_from)/vocabulary_size_from, 4)*100))


# 回答中词表的映射
concat_to = ' '.join(short_answers+answer_test).split()
vocabulary_size_to = len(list(set(concat_to)))
data_to, count_to, dictionary_to, rev_dictionary_to = build_dataset(concat_to, vocabulary_size_to)
print('词的总数量', vocabulary_size_to)
print('出现频率最高的6个词', count_to[4:10])
print('id和词之间的对应:', data_to[:10], [rev_dictionary_to[i] for i in data_to[:10]])
print('词表的大小:', len(dictionary_to))
print("词表中词的利用比:{}".format(round(len(dictionary_to)/vocabulary_size_to, 4)*100))


GO = dictionary_from['GO']
PAD = dictionary_from['PAD']
EOS = dictionary_from['EOS']
UNK = dictionary_from['UNK']

# 给每个回答加结束标志
for i in range(len(short_answers)):
    short_answers[i] += ' EOS'


# 定义模型
class Chatbot:
    def __init__(self, size_layer, num_layers, embedded_size,
                 from_dict_size, to_dict_size, learning_rate, batch_size):

        # 1.定义基本的RNN单元
        def cells(reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        # 2.定义占位符
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.Y_seq_len = tf.placeholder(tf.int32, [None])
        batch_size = tf.shape(self.X)[0]

        # 初始化词嵌入的向量
        encoder_embeddings = tf.Variable(tf.random_uniform([from_dict_size, embedded_size], -1, 1))
        # decoder_embeddings = tf.Variable(tf.random_uniform([to_dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        main = tf.strided_slice(self.X, [0, 0], [batch_size, -1], [1, 1])

        # 解码的输入 也是问题  然后进行词嵌入
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)  # 在每一行的开始加一个GO
        decoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, decoder_input)

        # 定义编码的模型结构 多层RNN   这里面的输入: 模型，问题的词向量, 问题的长度, 类型
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        _, last_state = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded,
                                          sequence_length=self.X_seq_len,
                                          dtype=tf.float32)

        # 定义解码的模型结构  多层RNN
        # 这里面的输入: 模型，问题(这里的问题多加了go字符)的词向量, 问题的长度, 类型, 这里的初始化直接用编码的输出初始化
        with tf.variable_scope("decoder"):
            rnn_cells_dec = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells_dec, decoder_embedded,
                                           sequence_length=self.X_seq_len,
                                           initial_state=last_state,
                                           dtype=tf.float32)     # 输入为decoder_embedded  用last_state初始化状态

        self.logits = tf.layers.dense(outputs, to_dict_size)   # 将输出 压缩为列为回答词表的长度

        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        # eg:
        # a = tf.sequence_mask([1, 2, 3], 5)
        # print(sess.run(a))
        # 输出：
        # [[ True False False False False]
        #  [ True  True False False False]
        #  [ True  True  True False False]]
        # 解释，最后面是5说的是有5列元素  前面有三个元素，依次代表每行从头开始有几个True

        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                     targets=self.Y,
                                                     weights=masks)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        y_t = tf.argmax(self.logits, axis=2)
        y_t = tf.cast(y_t, tf.int32)
        self.prediction = tf.boolean_mask(y_t, masks)
        mask_label = tf.boolean_mask(self.Y, masks)
        correct_pred = tf.equal(self.prediction, mask_label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 定义一些参数
size_layer = 256
num_layers = 2
embedded_size = 128
learning_rate = 0.001
batch_size = 16
epoch = 20


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Chatbot(size_layer, num_layers, embedded_size, len(dictionary_from),
                len(dictionary_to), learning_rate,batch_size)
sess.run(tf.global_variables_initializer())


def str_idx(corpus, dic):
    # 来一条数据 将其转为id序列
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k, UNK))
        X.append(ints)
    return X


# 将训练集和测试集的 问题,回答 转为id序列
X = str_idx(short_questions, dictionary_from)
Y = str_idx(short_answers, dictionary_to)
X_test = str_idx(question_test, dictionary_from)
Y_test = str_idx(answer_test, dictionary_from)


# 获取问题和回答的长度
maxlen_question = max([len(x) for x in X]) * 2
maxlen_answer = max([len(y) for y in Y]) * 2


def pad_sentence_batch(sentence_batch, pad_int, maxlen):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
    return padded_seqs, seq_lens


for i in range(epoch):
    total_loss, total_accuracy = 0, 0
    X, Y = shuffle(X, Y)
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
    print('epoch: %d, avg loss: %f, avg accuracy: %f'%(i+1, total_loss, total_accuracy))


for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:', ' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]]))
    print('REAL ANSWER:', ' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0, 1, 2, 3]]))
    print('PREDICTED ANSWER:', ' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0, 1, 2, 3]]), '\n')


batch_x, seq_x = pad_sentence_batch(X_test[:batch_size], PAD, maxlen_answer)
batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD, maxlen_answer)
predicted = sess.run(tf.argmax(model.logits, 2), feed_dict={model.X: batch_x, model.X_seq_len:seq_x})


for i in range(len(batch_x)):
    print('row %d'%(i+1))
    print('QUESTION:', ' '.join([rev_dictionary_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]]))
    print('REAL ANSWER:', ' '.join([rev_dictionary_to[n] for n in batch_y[i] if n not in[0, 1, 2, 3]]))
    print('PREDICTED ANSWER:', ' '.join([rev_dictionary_to[n] for n in predicted[i] if n not in[0, 1, 2, 3]]), '\n')


