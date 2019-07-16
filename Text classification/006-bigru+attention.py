"""

@file   : 006-bigru+attention.py

@author : xiaolu

@time   : 2019-07-16

"""
import glob
import json
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf


def read_data(p):
    data = []
    with open(p, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            data.append(line)
    return data


def process_data(datas):
    total_str = ''.join(datas)
    vocab = list(set(list(total_str)))
    print("词的个数:", len(vocab))

    # pad 0, unk 1
    vocab2id = {}   # 词表
    vocab2id['pad'] = 0
    vocab2id['unk'] = 1
    for i, v in enumerate(vocab):
        vocab2id[v] = i+2


    # 找出文本中最常的 然后将其他不够长度的进行padding
    max_len = 0
    for _ in datas:
        temp = len(_)
        if temp > max_len:
            max_len = temp
    print("样本中长度最长的为:", max_len)

    # 这里稍加处理，即文本特别短的 也就是长度小于5的，我们将其复制一次
    extend_corpus = []
    for d in datas:
        if len(d) < 5:
            d += d
            extend_corpus.append(d)
        else:
            extend_corpus.append(d)

    data_num = []
    for d in extend_corpus:
        temp = [vocab2id.get(c, 1) for c in d]
        if len(temp) < max_len:
            temp += [0] * (max_len - len(temp))
        data_num.append(temp)
    print(len(data_num))

    return data_num, vocab2id


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    # 定义普通的attention
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def build_model(vocab_size):
    # 建立模型
    # 1.定义占位符
    X = tf.placeholder(tf.int32, [None, 49])
    Y = tf.placeholder(tf.float32, [None, 5])

    # 2. 进行词嵌入
    embedded_size = 128
    embeddings = tf.Variable(tf.random_uniform([vocab_size, embedded_size], -1, 1))
    embedded = tf.nn.embedding_lookup(embeddings, X)

    # 3. 构建多层RNN
    cell = []
    for i in range(2):
        cell.append(tf.contrib.rnn.LSTMCell(64))

    rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cell)
    outputs, _ = tf.nn.dynamic_rnn(rnn_cells, embedded, dtype=tf.float32)

    # 4. 再加一点普通注意力
    ATTENTION_SIZE = 64
    attention_output, alphas = attention(outputs, ATTENTION_SIZE, return_alphas=True)

    pred = tf.contrib.layers.fully_connected(attention_output, 5)

    # 5. 定义损失
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

    # 6. 定义优化器
    learning_rate = 0.01
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    batch_size = 128
    for i in range(10):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 1
            while step * batch_size < len(x):
                batch_x, batch_y = x[batch_size* (step-1): batch_size * step], \
                                   y[batch_size* (step-1): batch_size * step]

                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})

                print("Iter:{}, acc:{}, loss: {}".format(step * batch_size, acc, loss))
                #
                # display_step = 10
                # if step % display_step == 0:
                #     acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
                #     loss = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
                #
                #     print("Iter:{}, acc:{}, loss: {}".format(step*batch_size, acc, loss))
            step += 1

        print('Finish')


if __name__ == '__main__':
    # 是否需要对数据进行预处理
    process_sign = True

    if process_sign:
        path_list = glob.glob('./data/*.data')
        datas = []
        labels = []
        for p in path_list:
            data_temp = read_data(p)
            label_temp = []
            label_temp.extend([str(int(p[-7])-1)] * len(data_temp))
            datas.extend(data_temp)
            labels.extend(label_temp)

        print(len(datas))    # 271826
        print(len(labels))   # 271826
        print(datas[:10])
        print(labels)

        # 整理数据
        data_num, vocab2id = process_data(datas)
        x = data_num
        y = labels
        json.dump([x, y, vocab2id], open('process_data.json', 'w'))
        x = np.array(x)
        y = to_categorical(y)

    else:
        x, y, vocab2id = json.load(open('process_data.json', 'r'))
        x = np.array(x)
        y = to_categorical(y)
        print(y)

        print(x.shape)
        print(y.shape)

    # 到此数据整理完毕
    vocab_size = len(vocab2id)
    build_model(vocab_size)

