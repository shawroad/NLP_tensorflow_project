"""

@file  : 003-LSTM-2path.py

@author: xiaolu

@time  : 2019-08-22

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
# tf.compat.v1.random.set_random_seed(1234)

df = pd.read_csv('./dataset/GOOG-year.csv')
# print(df.head())

minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32'))
df_log = minmax.transform(df.iloc[:, 4:5]).astype('float32')
df_log = pd.DataFrame(df_log)
# print(df_log.head())

test_size = 30   # 倒数二十个作为测试集
simulation_size = 10

df_train = df_log.iloc[:-test_size]
df_test = df_log.iloc[-test_size:]
print(df.shape, df_train.shape, df_test.shape)   # (252, 7) (222, 1) (30, 1)


class Model:
    def __init__(self, learning_rate, num_layers, size, size_layer, output_size, forget_bias=0.1):
        '''
        :param learning_rate:
        :param num_layers:
        :param size:
        :param size_layer:
        :param output_size:
        :param forget_bias:
        '''
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        with tf.variable_scope('forward', reuse=False):
            # 单个正向的LSTM
            self.X_forward = tf.placeholder(tf.float32, (None, None, size))
            self.hidden_layer_forward = tf.placeholder(tf.float32, (None, num_layers * 2 * size_layer))

            rnn_cells_forward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)],
                state_is_tuple=False
            )

            drop_forward = tf.contrib.rnn.DropoutWrapper(
                rnn_cells_forward, output_keep_prob=forget_bias
            )

            self.outputs_forward, self.last_state_forward = tf.nn.dynamic_rnn(
                drop_forward,
                self.X_forward,
                initial_state=self.hidden_layer_forward,
                dtype=tf.float32
            )

        with tf.variable_scope('backward', reuse=False):
            self.X_backward = tf.placeholder(tf.float32, (None, None, size))
            self.hidden_layer_backward = tf.placeholder(tf.float32, (None, num_layers * 2 * size_layer))

            rnn_cells_backward = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(size_layer) for _ in range(num_layers)],
                state_is_tuple=False
            )

            drop_backward = tf.contrib.rnn.DropoutWrapper(
                rnn_cells_backward, output_keep_prob=forget_bias
            )

            self.outputs_backward, self.last_state_backward = tf.nn.dynamic_rnn(
                drop_backward,
                self.X_backward,
                initial_state=self.hidden_layer_backward,
                dtype=tf.float32
            )

        # 将两个方向融合
        self.outputs = self.outputs_backward - self.outputs_forward
        self.Y = tf.placeholder(tf.float32, (None, output_size))

        self.logits = tf.layers.dense(self.outputs[-1], output_size)

        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


def calculate_accuracy(real, predict):
    # 计算一个类似准确率的东西
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
    return percentage * 100


def anchor(signal, weight):
    # 将预测的曲线变平滑
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


num_layers = 1
size_layer = 128
timestamp = 5
epoch = 300
dropout_rate = 0.8
future_day = test_size
learning_rate = 0.01


def forecast():
    # 训练加预测
    tf.reset_default_graph()
    model = Model(
        learning_rate, num_layers, df_log.shape[1], size_layer, df_log.shape[1], dropout_rate
    )
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()
    
    for i in range(epoch):
        init_value_forward = np.zeros((1, num_layers * 2 * size_layer))
        init_value_backward = np.zeros((1, num_layers * 2 * size_layer))
        
        total_loss, total_acc = [], []
        
        for k in range(0, df_train.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x_forward = np.expand_dims(
                df_train.iloc[k: index, :].values, axis=0
            )
            batch_x_backward = np.expand_dims(
                np.flip(df_train.iloc[k: index, :].values, axis=0), axis=0
            )
            batch_y = df_train.iloc[k + 1: index + 1, :].values
            logits, last_state_forward, last_state_backward, _, loss = sess.run(
                [
                    model.logits,
                    model.last_state_forward,
                    model.last_state_backward,
                    model.optimizer,
                    model.cost,
                ],
                feed_dict={
                    model.X_forward: batch_x_forward,
                    model.X_backward: batch_x_backward,
                    model.Y: batch_y,
                    model.hidden_layer_forward: init_value_forward,
                    model.hidden_layer_backward: init_value_backward,
                },
            )
            init_value_forward = last_state_forward
            init_value_backward = last_state_backward
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
            print('当前epoch:{}, 当前步:{}, 损失:{}, 准确率:{}'.format(i, k, np.mean(total_loss), np.mean(total_acc)))

    future_day = test_size
    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp
    init_value_forward = np.zeros((1, num_layers * 2 * size_layer))
    init_value_backward = np.zeros((1, num_layers * 2 * size_layer))

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        batch_x_forward = np.expand_dims(
            df_train.iloc[k: k + timestamp, :], axis=0
        )
        batch_x_backward = np.expand_dims(
            np.flip(df_train.iloc[k: k + timestamp, :].values, axis=0), axis=0
        )
        out_logits, last_state_forward, last_state_backward = sess.run(
            [
                model.logits,
                model.last_state_forward,
                model.last_state_backward,
            ],
            feed_dict={
                model.X_forward: batch_x_forward,
                model.X_backward: batch_x_backward,
                model.hidden_layer_forward: init_value_forward,
                model.hidden_layer_backward: init_value_backward,
            },
        )
        init_value_forward = last_state_forward
        init_value_backward = last_state_backward
        output_predict[k + 1: k + timestamp + 1, :] = out_logits

    if upper_b != df_train.shape[0]:
        batch_x_forward = np.expand_dims(df_train.iloc[upper_b:, :], axis=0)
        batch_x_backward = np.expand_dims(
            np.flip(df_train.iloc[upper_b:, :].values, axis=0), axis=0
        )
        out_logits, last_state_forward, last_state_backward = sess.run(
            [model.logits, model.last_state_forward, model.last_state_backward],
            feed_dict={
                model.X_forward: batch_x_forward,
                model.X_backward: batch_x_backward,
                model.hidden_layer_forward: init_value_forward,
                model.hidden_layer_backward: init_value_backward,
            },
        )
        init_value_forward = last_state_forward
        init_value_backward = last_state_backward
        output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days=1))

    init_value_forward = last_state_forward
    init_value_backward = last_state_backward

    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i:-future_day + i]
        o_f = np.flip(o, axis=0)
        out_logits, last_state_forward, last_state_backward = sess.run(
            [
                model.logits,
                model.last_state_forward,
                model.last_state_backward,
            ],
            feed_dict={
                model.X_forward: np.expand_dims(o, axis=0),
                model.X_backward: np.expand_dims(o_f, axis=0),
                model.hidden_layer_forward: init_value_forward,
                model.hidden_layer_backward: init_value_backward,
            },
        )
        init_value_forward = last_state_forward
        init_value_backward = last_state_backward
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days=1))

    output_predict = minmax.inverse_transform(output_predict)
    deep_future = anchor(output_predict[:, 0], 0.3)

    return deep_future[-test_size:]


results = []
for i in range(simulation_size):
    print('simulation %d'%(i + 1))
    results.append(forecast())


accuracies = [calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]

plt.figure(figsize=(15, 5))
for no, r in enumerate(results):
    plt.plot(r, label='forecast %d'%(no + 1))
plt.plot(df['Close'].iloc[-test_size:].values, label='true trend', c='black')
plt.legend()
plt.title('average accuracy: %.4f'%(np.mean(accuracies)))
plt.show()
