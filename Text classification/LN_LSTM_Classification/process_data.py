"""

@file  : process_data.py

@author: xiaolu

@time  : 2019-07-16

"""
import glob
import json
import tensorflow as tf
from keras.utils import to_categorical
import numpy as np
from Model_Lstm_LN import Model


def read_data(p):
    data = []
    with open(p, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            data.append(line)
    return data


def process_data(datas):
    # 预处理数据
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
    seq_len = []
    for d in extend_corpus:
        temp = [vocab2id.get(c, 1) for c in d]
        seq_len.append(len(temp))
        if len(temp) < max_len:
            temp += [0] * (max_len - len(temp))
        data_num.append(temp)
    print(len(data_num))

    return data_num, seq_len, vocab2id


if __name__ == '__main__':
    # 是否需要对数据进行预处理
    process_sign = True

    if process_sign:
        path_list = glob.glob('./data/*.data')
        datas = []
        labels = []
        i = 0
        for p in path_list:
            data_temp = read_data(p)
            label_temp = []
            label_temp.extend([i] * len(data_temp))
            datas.extend(data_temp)
            labels.extend(label_temp)
            i += 1

        print(len(datas))  # 271826
        print(len(labels))  # 271826
        print(labels[0:len(datas):10000])
        print(datas[:10])

        # 整理数据
        data_num, seq_len, vocab2id = process_data(datas)
        x = data_num
        y = labels
        json.dump([x, y, seq_len, vocab2id], open('process_data1.json', 'w'))
        x = np.array(x)
        y = to_categorical(y)

    else:
        x, y, seq_len, vocab2id = json.load(open('process_data1.json', 'r'))
        x = np.array(x)
        y = to_categorical(y)
        print(y)

        print(x.shape)
        print(y.shape)

    # 到此数据整理完毕
    vocab_size = len(vocab2id)

    # 定义模型参数
    num_layers = 2
    size_layer = 128
    learning_rate = 0.01
    input_dim = 49
    output_dim = 5

    sess = tf.Session()

    model = Model(num_layers, size_layer, input_dim, output_dim, learning_rate)

    batch_size = 128
    for i in range(10):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 1
            while step * batch_size < len(x):
                batch_x, batch_y = x[batch_size * (step-1): batch_size * step], \
                                   y[batch_size * (step-1): batch_size * step]

                sess.run(model.optimizer, feed_dict={model.X: batch_x, model.Y: batch_y})
                acc = sess.run(model.accuracy, feed_dict={model.X: batch_x, model.Y: batch_y})
                loss = sess.run(model.cost, feed_dict={model.X: batch_x, model.Y: batch_y})

                print("Iter:{}, acc:{}, loss: {}".format(step * batch_size, acc, loss))

            step += 1

        print('Finish')

