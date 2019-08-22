"""

@file  : 001-LSTM.py

@author: xiaolu

@time  : 2019-08-20

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
        :param learning_rate: 学习率
        :param num_layers: 多少层lstm
        :param size:
        :param size_layer: 每步的输出维度
        :param output_size:
        :param forget_bias: dropout rate
        '''
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)  # 不需要输出每步的状态

        # 多层lstm
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(size_layer) for _ in range(num_layers)],
                                                state_is_tuple=False)

        self.X = tf.placeholder(tf.float32, (None, None, size))
        self.Y = tf.placeholder(tf.float32, (None, output_size))

        drop = tf.contrib.rnn.DropoutWrapper(
            rnn_cells, output_keep_prob=forget_bias
        )

        # 构造初始化的隐状态
        self.hidden_layer = tf.placeholder(tf.float32, (None, num_layers*2*size_layer))

        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop,
            self.X,
            initial_state=self.hidden_layer,
            dtype=tf.float32
        )

        self.logits = tf.layers.dense(self.outputs[-1], output_size)  # 将最后的输出进行dense
        # 定义损失
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))

        # 定义优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize((self.cost))


def calculate_accuracy(real, predict):
    # 计算准确率
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))   # 类似均方误差然后出真实值　最后被１减
    return percentage * 100


# deep_future = anchor(output_predict[:, 0], 0.3)
def anchor(signal, weight):
    # 对数据平滑
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
    '''
    预测
    :return:
    '''
    tf.reset_default_graph()  # 重置原来图
    model = Model(learning_rate,
                  num_layers,
                  df_log.shape[1],
                  size_layer,
                  df_log.shape[1],
                  dropout_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()   # 将日期转为真正日期 然后拉成列表

    for i in range(epoch):
        init_value = np.zeros((1, num_layers * 2 * size_layer))  # 初始化隐状态　全零初始化
        total_loss, total_acc = [], []
        for k in range(0, df_train.shape[0] - 1, timestamp):  # 已知道五步
            index = min(k + timestamp, df_train.shape[0] - 1)
            batch_x = np.expand_dims(df_train.iloc[k: index, :].values, axis=0)  # 知道当前五步
            batch_y = df_train.iloc[k + 1: index + 1, :].values  # 往后移动一步

            logits, last_state, _, loss = sess.run([model.logits, model.last_state, model.optimizer, model.cost],
                                                   feed_dict={
                                                       model.X: batch_x,
                                                       model.Y: batch_y,
                                                       model.hidden_layer: init_value
                                                   })

            init_value = last_state
            total_loss.append(loss)
            total_acc.append(calculate_accuracy(batch_y[:, 0], logits[:, 0]))
            print('当前epoch:{}, 当前步:{}, 损失:{}, 准确率:{}'.format(i, k, np.mean(total_loss), np.mean(total_acc)))

    # 进行预测
    future_day = test_size  # 预测未来的几天
    output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
    output_predict[0] = df_train.iloc[0]
    upper_b = (df_train.shape[0] // timestamp) * timestamp

    init_value = np.zeros((1, num_layers * 2 * size_layer))  # 初始化的隐状态

    for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
        out_logits, last_state = sess.run([model.logits, model.last_state],
                                          feed_dict={model.X: np.expand_dims(df_train.iloc[k: k+timestamp], axis=0),
                                                     model.hidden_layer: init_value})
        init_value = last_state  # 上一步的最后一步的隐状态作为下一次预测的初始状态
        output_predict[k + 1: k+timestamp + 1] = out_logits

    if upper_b != df_train.shape[0]:   # 最后一点点可能不够一个timestamp, 所以我们特殊考虑
        out_logits, last_state = sess.run([model.logits, model.last_state],
                                          feed_dict={model.X: np.expand_dims(df_train.iloc[upper_b:], axis=0),
                                                     model.hidden_layer: init_value})

        output_predict[upper_b + 1: df_train.shape[0] + 1] = out_logits
        future_day -= 1
        date_ori.append(date_ori[-1] + timedelta(days=1))
        init_value = last_state

    # 需要往后预测future_day天
    for i in range(future_day):
        o = output_predict[-future_day - timestamp + i: -future_day + i]   # 往output_predict加了几步　我们需要预测将其填入
        out_logits, last_state = sess.run([model.logits, model.last_state],
                                          feed_dict={model.X: np.expand_dims(o, axis=0),
                                                     model.hidden_layer: init_value})

        init_value = last_state
        output_predict[-future_day + i] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(days=1))

    output_predict = minmax.inverse_transform(output_predict)  # 因为我们对数据进行了标准化 所以这里需要还原数据
    deep_future = anchor(output_predict[:, 0], 0.3)

    return deep_future[-test_size:]


results = []
for i in range(simulation_size):
    print('simulation %d'%(i + 1))
    results.append(forecast())


accuracies = [calculate_accuracy(df['Close'].iloc[-test_size:].values, r) for r in results]

plt.figure(figsize=(15, 5))
for no, r in enumerate(results):
    plt.plot(r, label='forcast %d'%(no + 1))

plt.plot(df['Close'].iloc[-test_size:].values, label='true trend', c='black')
plt.legend()
plt.title('average accuracy: %.4f'% (np.mean(accuracies)))
plt.show()