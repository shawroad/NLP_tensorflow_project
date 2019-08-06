"""

@file  : 001-seq2seq+luongAttention.py

@author: xiaolu

@time  : 2019-07-31

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.preprocessing.sequence import pad_sequences
import os
import pickle


def load_vocab(path):
    # 读取词表
    with open(path, 'r') as fr:
        vocab = fr.readlines()
        vocab = [w.strip('\n') for w in vocab]
    return vocab


# 分别读取中英文字典   我们这里只保留的英文和中文常用的前20000个词，所有就有部分词是没有的，我们用UNK代替。
vocab_ch = load_vocab('data/vocab.ch')
vocab_en = load_vocab('data/vocab.en')
print(len(vocab_ch), vocab_ch[:20])    # 查看中文字典的前20个词
print(len(vocab_en), vocab_en[:20])    # 查看英文字典的前20个词
# '<unk>': 代表不存在的词, '<s>':代表一句话的起始位置, '</s>':代表一句话的终止位置


# 将单词和id进行映射  建立中文, 英文词典
word2id_ch = {w: i for i, w in enumerate(vocab_ch)}
id2word_ch = {i: w for i, w in enumerate(vocab_ch)}
word2id_en = {w: i for i, w in enumerate(vocab_en)}
id2word_en = {i: w for i, w in enumerate(vocab_en)}


# 加载训练集、验证集、测试集数据，计算中英文数据对应的最大序列长度，
# 并根据mode对相应数据进行padding
def load_data(path, word2id):
    with open(path, 'r') as fr:
        lines = fr.readlines()
        sentences = [line.strip('\n').split(' ') for line in lines]
        sentences = [[word2id['<s>']] + [word2id[w] for w in sentence] + [word2id['</s>']]
                     for sentence in sentences]
        lens = [len(sentence) for sentence in sentences]  # 获取的是每句话的长度
        maxlen = np.max(lens)
        return sentences, lens, maxlen   # 得到每个句子转化为对应的id序列，以及对应的长度和最大的长度


# train: training, no beam search, calculate loss
# eval: no training, no beam search, calculate loss
# infer: no training, beam search, calculate bleu
mode = 'train'   # 确定对那种数据集进行操作

# 加载三种类型的数据
train_ch, len_train_ch, maxlen_train_ch = load_data('./data/train.ch', word2id_ch)
train_en, len_train_en, maxlen_train_en = load_data('./data/train.en', word2id_en)
dev_ch, len_dev_ch, maxlen_dev_ch = load_data('./data/dev.ch', word2id_ch)
dev_en, len_dev_en, maxlen_dev_en = load_data('./data/dev.en', word2id_en)
test_ch, len_test_ch, maxlen_test_ch = load_data('./data/test.ch', word2id_ch)
test_en, len_test_en, maxlen_test_en = load_data('./data/test.en', word2id_en)


# 返回一句话中最常的长度
maxlen_ch = np.max([maxlen_train_ch, maxlen_dev_ch, maxlen_test_ch])
maxlen_en = np.max([maxlen_train_en, maxlen_dev_en, maxlen_test_en])
print(maxlen_ch, maxlen_en)    # 中文最大长度62， 英文最大长度也是62


# 对数据进行填充  填充方式:在尾部填充'</s>'
if mode == 'train':
    train_ch = pad_sequences(train_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    train_en = pad_sequences(train_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print(train_ch.shape, train_en.shape)   # (100000, 62), (100000, 62)
elif mode == 'eval':
    dev_ch = pad_sequences(dev_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    dev_en = pad_sequences(dev_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print(dev_ch.shape, dev_en.shape)
elif mode == 'infer':
    test_ch = pad_sequences(test_ch, maxlen=maxlen_ch, padding='post', value=word2id_ch['</s>'])
    test_en = pad_sequences(test_en, maxlen=maxlen_en, padding='post', value=word2id_en['</s>'])
    print(test_ch.shape, test_en.shape)


# 定义四个placeholder, 对输入进行嵌入
X = tf.placeholder(tf.int32, [None, maxlen_ch])  # 中文的输入
X_len = tf.placeholder(tf.int32, [None])    # 每个中文的长度
Y = tf.placeholder(tf.int32, [None, maxlen_en])
Y_len = tf.placeholder(tf.int32, [None])
Y_in = Y[:, :-1]   # 定义输入
Y_out = Y[:, 1:]   # 定义输出

# 初始化
k_initializer = tf.contrib.layers.xavier_initializer()
e_initializer = tf.random_uniform_initializer(-1.0, 1.0)

embedding_size = 512
hidden_size = 512

if mode == 'train':
    batch_size = 128
else:
    batch_size = 16

with tf.variable_scope('embedding_X'):
    embeddings_X = tf.get_variable('weights_X', [len(word2id_ch), embedding_size], initializer=e_initializer)
    embedded_X = tf.nn.embedding_lookup(embeddings_X, X)  # batch_size, seq_len, embedding_size

with tf.variable_scope('embedding_Y'):
    embeddings_Y = tf.get_variable('weights_Y', [len(word2id_en), embedding_size], initializer=e_initializer)
    embedded_Y = tf.nn.embedding_lookup(embeddings_Y, Y_in)  # batch_size, seq_len, embedding_size


# define encoder
def single_cell(mode=mode):
    # 定义单个单元
    if mode == 'train':
        keep_prob = 0.8
    else:
        keep_prob = 1.0
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
    return cell


def multi_cells(num_layers):
    # 多层的lstm
    cells = []
    for i in range(num_layers):
        cell = single_cell()
        cells.append(cell)
    return tf.nn.rnn_cell.MultiRNNCell(cells)


with tf.variable_scope('encoder'):
    num_layers = 1
    fw_cell = multi_cells(num_layers)
    bw_cell = multi_cells(num_layers)
    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded_X, dtype=tf.float32,
                                                           sequence_length=X_len)
    # fw: batch_size, seq_len, hidden_size
    # bw: batch_size, seq_len, hidden_size
    print('*'*100 + '\n', bi_outputs)   # (shape=(?, 62, 512), shape=(?, 62, 512))

    encoder_outputs = tf.concat(bi_outputs, axis=-1)
    print('*'*100 + '\n', encoder_outputs.shape)  # shape=(?, 62, 1024)  batch_size, seq_len, 2*hidden_size

    # tuple(fw & bw), 2 tuple(c & h), batch_size, hidden_size
    print('*'*100 + '\n', bi_state)    # ((c, h), (c, h))  因为我们这里使用的是两层的lstm

    encoder_state = []
    for i in range(num_layers):
        encoder_state.append(bi_state[0][i])  # forward
        encoder_state.append(bi_state[1][i])  # backward

    encoder_state = tuple(encoder_state)
    print('*'*100)

    for i in range(len(encoder_state)):
        print(i, encoder_state[i])

# exit()


# define decoder
with tf.variable_scope('decoder'):
    beam_width = 10
    memory = encoder_outputs

    if mode == 'infer':
        # 推断
        memory = tf.contrib.seq2seq.tile_batch(memory, beam_width)
        X_len_ = tf.contrib.seq2seq.tile_batch(X_len, beam_width)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
        bs = batch_size * beam_width
    else:
        bs = batch_size
        X_len_ = X_len

    attention = tf.contrib.seq2seq.LuongAttention(hidden_size, memory, X_len_, scale=True)  # multiplicative
    # attention = tf.contrib.seq2seq.BahdanauAttention(hidden_size, memory, X_len_, normalize=True) # additive
    cell = multi_cells(num_layers * 2)   # 定义解码的网络单元
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention, hidden_size, name='attention')  # 加注意力的解码
    decoder_initial_state = cell.zero_state(bs, tf.float32).clone(cell_state=encoder_state)  # 将编码后的状态初始化给解码的初始状态

    with tf.variable_scope('projected'):
        output_layer = tf.layers.Dense(len(word2id_en), use_bias=False, kernel_initializer=k_initializer)

    if mode == 'infer':
        # 推断过程
        start = tf.fill([batch_size], word2id_en['<s>'])   # 对于推断, 我们在句子的最前面添加<s>
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell, embeddings_Y, start, word2id_en['</s>'],
                                                       decoder_initial_state, beam_width, output_layer)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                            output_time_major=True,
                                                                            maximum_iterations=2 * tf.reduce_max(X_len))
        sample_id = outputs.predicted_ids
    else:
        # 训练过程
        helper = tf.contrib.seq2seq.TrainingHelper(embedded_Y, [maxlen_en - 1 for b in range(batch_size)])
        decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state, output_layer)

        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                            output_time_major=True)
        logits = outputs.rnn_output
        logits = tf.transpose(logits, (1, 0, 2))
        print(logits)   # shape=(128, ?, 20003)


if mode != 'infer':
    with tf.variable_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_out, logits=logits)
        mask = tf.sequence_mask(Y_len, tf.shape(Y_out)[1], tf.float32)
        loss = tf.reduce_sum(loss * mask) / batch_size

if mode == 'train':
    learning_rate = tf.Variable(0.0, trainable=False)
    params = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, params), 5.0)   # 梯度裁剪
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads, params))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

if mode == 'train':
    saver = tf.train.Saver()
    OUTPUT_DIR = 'model_diy'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    # 收集损失标量
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(OUTPUT_DIR)

    epochs = 20
    for e in range(epochs):
        total_loss = 0
        total_count = 0

        # 让学习率逐渐减小
        start_decay = int(epochs * 2 / 3)
        if e <= start_decay:
            lr = 1.0
        else:
            decay = 0.5 ** (int(4 * (e - start_decay) / (epochs - start_decay)))
            lr = 1.0 * decay

        sess.run(tf.assign(learning_rate, lr))   # 将lr赋值给learning_rate

        # 把数据打乱
        train_ch, len_train_ch, train_en, len_train_en = shuffle(train_ch, len_train_ch, train_en, len_train_en)

        for i in range(train_ch.shape[0] // batch_size):
            X_batch = train_ch[i * batch_size: i * batch_size + batch_size]
            X_len_batch = len_train_ch[i * batch_size: i * batch_size + batch_size]
            Y_batch = train_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = len_train_en[i * batch_size: i * batch_size + batch_size]
            Y_len_batch = [l - 1 for l in Y_len_batch]

            feed_dict = {X: X_batch, Y: Y_batch, X_len: X_len_batch, Y_len: Y_len_batch}
            _, ls_ = sess.run([optimizer, loss], feed_dict=feed_dict)

            total_loss += ls_ * batch_size
            total_count += np.sum(Y_len_batch)

            if i > 0 and i % 100 == 0:
                writer.add_summary(sess.run(summary, feed_dict=feed_dict),
                                   e * train_ch.shape[0] // batch_size + i)
                writer.flush()

        print('Epoch %d lr %.3f perplexity %.2f' % (e, lr, np.exp(total_loss / total_count)))
        saver.save(sess, os.path.join(OUTPUT_DIR, 'nmt'))
