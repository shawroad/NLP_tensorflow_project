"""

@file  : 009-cnn_seq2seq.py

@author: xiaolu

@time  : 2019-08-27

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

sns.set()
tf.random.set_random_seed(1234)

df = pd.read_csv('./dataset/GOOG-year.csv')

# 标准化
minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))
df_log = minmax.transform(df.iloc[:, 4:5]).astype('float32')
df_log = pd.DataFrame(df_log)

test_size = 30   # 倒数二十个作为测试集
simulation_size = 10

df_train = df_log.iloc[:-test_size]
df_test = df_log.iloc[-test_size:]
print(df.shape, df_train.shape, df_test.shape)   # (252, 7) (222, 1) (30, 1)


def encoder_block(inp, n_hidden, filter_size):
    # 编码的卷积块
    # 对序列进行卷积 先要对序列进行扩充
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(inp,
                 [[0, 0], [(filter_size[0] - 1) // 2, (filter_size[0] - 1) // 2],
                  [0, 0], [0, 0]],
                 )
    conv = tf.layers.conv2d(
        inp, n_hidden, filter_size, padding='VALID', activation=None
    )

    # 然后有把扩充的那个维度压没
    conv = tf.squeeze(conv, 2)
    return conv


def decoder_block(inp, n_hidden, filter_size):
    # 解码的卷积块
    inp = tf.expand_dims(inp, 2)
    inp = tf.pad(inp, [[0, 0], [filter_size[0] - 1, 0], [0, 0], [0, 0]])
    conv = tf.layers.conv2d(
        inp, n_hidden, filter_size, padding='VALID', activation=None
    )
    conv = tf.squeeze(conv, 2)
    return conv


def glu(x):
    return tf.multiply(x[:, :, : tf.shape(x)[2] // 2], tf.sigmoid(x[:, :, tf.shape(x)[2] // 2:]))


def layer(inp, conv_block, kernel_width, n_hidden, residual=None):
    # 定义一个块
    # z = layer(encoder_embedded, encoder_block, kernel_size, size_layer * 2, encoder_embedded)
    z = conv_block(inp, n_hidden, (kernel_width, 1))
    return glu(z) + (residual if residual is not None else 0)  # 是否要带残差


class Model:
    def __init__(self, learning_rate, num_layers, size, size_layer, output_size,
                 kernel_size=3, n_attn_heads=16, dropout=0.9):
        '''
        :param learning_rate:
        :param num_layers:
        :param size:
        :param size_layer:
        :param output_size:
        :param kernel_size:
        :param n_attn_heads:
        :param dropout:
        '''
        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

        # 通过一个dense
        encoder_embedded = tf.layers.dense(self.X, size_layer)  # 相当于embedding

        e = tf.identity(encoder_embedded)  # 相当于备份一份

        # 编码
        for i in range(num_layers):
            z = layer(encoder_embedded, encoder_block, kernel_size, size_layer * 2, encoder_embedded)
            z = tf.nn.dropout(z, keep_prob=dropout)
            encoder_embedded = z

        encoder_output, output_memory = z, z + e

        g = tf.identity(encoder_embedded)

        # 解码
        for i in range(num_layers):
            attn_res = h = layer(encoder_embedded, decoder_block, kernel_size,
                                 size_layer * 2, residual=tf.zeros_like(encoder_embedded))

            C = []
            for j in range(n_attn_heads):
                h_ = tf.layers.dense(h, size_layer // n_attn_heads)
                g_ = tf.layers.dense(g, size_layer // n_attn_heads)
                zu_ = tf.layers.dense(encoder_output, size_layer // n_attn_heads)

                ze_ = tf.layers.dense(output_memory, size_layer // n_attn_heads)

                d = tf.layers.dense(h_, size_layer // n_attn_heads) + g_
                dz = tf.matmul(d, tf.transpose(zu_, [0, 2, 1]))
                a = tf.nn.softmax(dz)
                c_ = tf.matmul(a, ze_)
                C.append(c_)

            c = tf.concat(C, 2)
            h = tf.layers.dense(attn_res + c, size_layer)
            h = tf.nn.dropout(h, keep_prob=dropout)
            encoder_embedded = h

        encoder_embedded = tf.sigmoid(encoder_embedded[-1])
        self.logits = tf.layers.dense(encoder_embedded, output_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


num_layers = 1
size_layer = 128
timestamp = test_size
epoch = 300
dropout_rate = 0.7
future_day = test_size
learning_rate = 1e-3


def forecast():
    tf.reset_default_graph()
    model = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1],
        dropout=dropout_rate
    )
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

    for i in range(epoch):
        init_value = np.zeros((1, num_layers * 2 * size_layer))
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(
                df_train.iloc[k: index, :].values, axis=0
            )
            batch_y = df_train.iloc[k + 1: index + 1, :].values
            logits, _, loss = sess.run(
                [model.logits, model.optimizer, model.cost],
                feed_dict={model.X: batch_x, model.Y: batch_y},
            )
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
            print('当前epoch:{}, 当前步:{}, 损失:{}, 准确率:{}'.format(i, k, np.mean(total_loss), np.mean(total_acc)))

    future_day = test_size
    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits = sess.run(
            model.logits,
            feed_dict={
                model.X: np.expand_dims(
                    df_train.iloc[k: k + timestamp], axis=0
                )
            },
        )
        output_predict[k + 1: k + timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:
        out_logits = sess.run(
            model.logits,
            feed_dict={
                model.X: np.expand_dims(df_train.iloc[upper_b:], axis=0)
            },
        )
        output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days=1))

    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        out_logits = sess.run(
            model.logits,
            feed_dict={
                model.X: np.expand_dims(o, axis=0)
            },
        )
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days=1))

    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.3)

    return deep_future[-test_size:]


results = []
for i in range(simulation_size):
    print('simulation %d' % (i + 1))
    results.append(forecast())


accuracies = [calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]

plt.figure(figsize=(15, 5))
for no, r in enumerate(results):
    plt.plot(r, label='forecast %d' % (no + 1))
plt.plot(df['Close'].iloc[-test_size:].values, label='true trend', c='black')
plt.legend()
plt.title('average accuracy: %.4f' % (np.mean(accuracies)))
plt.show()
