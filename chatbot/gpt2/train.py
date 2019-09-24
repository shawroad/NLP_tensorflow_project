"""

@file  : train.py

@author: xiaolu

@time  : 2019-09-23

"""
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import re
import time
import collections
import os
import json
from tensorflow.contrib.training import HParams
import model_GPT2


class Chatbot:
    def __init__(self):
        # 1. 定义输入
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None, None])

        self.X_seq_len = tf.count_nonzero(self.X, 1, dtype=tf.int32)
        self.Y_seq_len = tf.count_nonzero(self.Y, 1, dtype=tf.int32)

        batch_size = tf.shape(self.X)[0]
        main = tf.strided_slice(self.Y, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO), main], 1)  # 解码的输入加起始标志
        # initializer = tf.initializers.random_normal(stddev=0.1)
        logits = tf.reduce_mean(model_GPT2.model(params, self.X)['logits'], axis=1)

        state_proj = tf.layers.Dense(params.n_embd)
        init_state = state_proj(logits)  # 构造解码的初始状态

        # 词嵌入
        embedding = tf.Variable(tf.random_uniform([len(id2vocab_to), params.n_embd], -1, 1))

        cell = tf.nn.rnn_cell.LSTMCell(params.n_embd)
        vocab_proj = tf.layers.Dense(len(id2vocab_to))

        # 解码
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(embedding, decoder_input),
            sequence_length=tf.to_int32(self.Y_seq_len)
        )

        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=init_state, h=init_state)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,
                                                  helper=helper,
                                                  initial_state=encoder_state,
                                                  output_layer=vocab_proj
                                                  )

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=tf.reduce_max(self.Y_seq_len)
        )
        # 推理
        # 贪婪搜索
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                          start_tokens=tf.tile(
                                                              tf.constant([GO], dtype=tf.int32),
                                                              [tf.shape(init_state)[0]]
                                                          ),
                                                          end_token=EOS
                                                          )

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=encoder_state,
            output_layer=vocab_proj
        )

        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=2*tf.reduce_max(self.X_seq_len)
        )

        self.training_logits = decoder_output.rnn_output
        self.predicting_ids = predicting_decoder_output.sample_id
        self.logits = decoder_output.sample_id
        masks = tf.sequence_mask(self.Y_seq_len, tf.reduce_max(self.Y_seq_len), dtype=tf.float32)
        self.cost = tf.contrib.seq2seq.sequence_loss(
            logits=self.training_logits,
            targets=self.Y,
            weights=masks
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

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


def clean_text(text):
    '''
    清洗语料
    :param text: 语料
    :return:
    '''
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



def str_idx(corpus, dic):
    '''
    将句子转为对应的id序列
    :param corpus:
    :param dic:
    :return:
    '''
    X = []
    for i in corpus:
        ints = []
        for k in i.split():
            ints.append(dic.get(k,UNK))
        X.append(ints)
    return X


def pad_sentence_batch(sentence_batch, pad_int, maxlen):
    '''
    对句子进行填充
    :param sentence_batch:
    :param pad_int:
    :param maxlen:
    :return:
    '''
    padded_seqs = []
    seq_lens = []
    max_sentence_len = maxlen
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(maxlen)
    return padded_seqs, seq_lens


def pad_sentence_batch_dynamic(sentence_batch, pad_int):
    padded_seqs = []
    seq_lens = []
    max_sentence_len = max([len(sentence) for sentence in sentence_batch])
    for sentence in sentence_batch:
        padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
        seq_lens.append(len(sentence))
    return padded_seqs, seq_lens


if __name__ == '__main__':
    # 1. 加载语料
    lines = open('./data/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
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

    # 2. 整理问题和回答
    for conv in convs:
        for i in range(len(conv) - 1):
            questions.append(id2line[conv[i]])
            answers.append(id2line[conv[i + 1]])

    # 3. 对语料进行简单清洗
    clean_questions = []
    for question in questions:
        clean_questions.append(clean_text(question))

    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))

    # 4. 只是为了看实验结果　　我们这里只选取超级短的句子
    min_line_length = 2
    max_line_length = 5
    short_questions_temp = []
    short_answers_temp = []

    i = 0
    for question in clean_questions:
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
            short_questions.append(short_questions_temp[i])
        i += 1

    question_test = short_questions[500:550]
    answer_test = short_answers[500:550]
    short_questions = short_questions[:500]
    short_answers = short_answers[:500]

    # 整理问题的词表
    concat_from = ' '.join(short_questions + question_test).split()
    vocabulary_size_from = len(list(set(concat_from)))
    data_from, count_from, vocab2id_from, id2vocab_from = build_dataset(concat_from, vocabulary_size_from)
    print("vocab from size:", vocabulary_size_from)
    print("Most common words:", concat_from[4:10])
    print("Sample data:", data_from[:10], [id2vocab_from[i] for i in data_from[:10]])
    print("filtered vocab size:", len(vocab2id_from))
    print("% of vocab used: {}%".format(round(len(vocab2id_from)/ vocabulary_size_from, 4) * 100))

    # 整理回答的词表
    concat_to = ' '.join(short_answers + answer_test).split()
    vocabulary_size_to = len(list(set(concat_to)))
    data_to, count_to, vocab2id_to, id2vocab_to = build_dataset(concat_to, vocabulary_size_from)
    print("vocab from size:", vocabulary_size_to)
    print("Most common words:", count_to[4:10])
    print("Sample data:", data_to[:10], [id2vocab_to[i] for i in data_to[:10]])
    print("filtered vocab size:", len(vocab2id_to))
    print("% of vocab used: {}%".format(round(len(vocab2id_to) / vocabulary_size_to, 4) * 100))

    GO = vocab2id_from['GO']
    PAD = vocab2id_from['PAD']
    EOS = vocab2id_from['EOS']
    UNK = vocab2id_from['UNK']

    # 给每个答案加结束标志
    for i in range(len(short_answers)):
        short_answers[i] += ' EOS'

    X = str_idx(short_questions, vocab2id_from)
    Y = str_idx(short_answers, vocab2id_to)
    X_test = str_idx(question_test, vocab2id_from)
    Y_test = str_idx(answer_test, vocab2id_to)

    maxlen_question = max([len(x) for x in X]) * 2
    maxlen_answer = max([len(y) for y in Y]) * 2

    # 超参数
    params = HParams(
        n_vocab=len(vocab2id_from),
        n_ctx=512,
        n_embd=256,
        n_head=8,
        n_layer=8,
    )

    batch_size = 16
    learning_rate = 1e-3
    epoch = 100

    tf.reset_default_graph()
    sess = tf.Session()
    model = Chatbot()
    sess.run(tf.global_variables_initializer())

    acc_list = []
    loss_list = []
    for i in range(epoch):
        total_loss, total_accuracy = 0, 0
        for k in range(0, len(short_questions), batch_size):
            index = min(k + batch_size, len(short_questions))
            batch_x, seq_x = pad_sentence_batch_dynamic(X[k: index], PAD)
            batch_y, seq_y = pad_sentence_batch_dynamic(Y[k: index], PAD)

            predicted, accuracy, loss, _ = sess.run([model.predicting_ids, model.accuracy, model.cost, model.optimizer],
                                                    feed_dict={
                                                        model.X: batch_x,
                                                        model.Y: batch_y
                                                    })
            total_loss += loss
            total_accuracy += accuracy
            print('epoch: %d, step: %d, loss: %f, accuracy: %f' % (i, k//batch_size, loss, accuracy))
        total_loss /= (len(short_questions) / batch_size)
        total_accuracy /= (len(short_questions) / batch_size)
        loss_list.append(total_loss)
        acc_list.append(total_accuracy)
        print('epoch: %d, avg loss: %f, avg accuracy: %f' % (i+1, total_loss, total_accuracy))

    # 保存准确率和损失
    final_list = []
    final_list.append(loss_list)
    final_list.append(acc_list)
    json.dump(final_list, open('acc_loss.json', 'w'))

    for i in range(len(batch_x)):
        print('row %d' % (i+1))
        print('QUESTION:', ' '.join([id2vocab_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]]))
        print('REAL ANSWER:', ' '.join([id2vocab_to[n] for n in batch_y[i] if n not in[0, 1, 2, 3]]))
        print('PREDICTED ANSWER:', ' '.join([id2vocab_to[n] for n in predicted[i] if n not in[0, 1, 2, 3]]), '\n')

    batch_x, seq_x = pad_sentence_batch(X_test[:batch_size], PAD, maxlen_question)
    batch_y, seq_y = pad_sentence_batch(Y_test[:batch_size], PAD, maxlen_answer)
    predicted = sess.run(model.predicting_ids, feed_dict={model.X:batch_x})

    for i in range(len(batch_x)):
        print('row %d'%(i+1))
        print('QUESTION:', ' '.join([id2vocab_from[n] for n in batch_x[i] if n not in [0, 1, 2, 3]]))
        print('REAL ANSWER:', ' '.join([id2vocab_to[n] for n in batch_y[i] if n not in[0, 1, 2, 3]]))
        print('PREDICTED ANSWER:', ' '.join([id2vocab_to[n] for n in predicted[i] if n not in[0, 1, 2, 3]]), '\n')
