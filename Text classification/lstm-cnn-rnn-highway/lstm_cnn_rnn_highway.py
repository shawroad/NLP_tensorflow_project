"""

@file  : lstm_cnn_rnn_highway.py

@author: xiaolu

@time  : 2019-07-24

"""

"""

@file  : lstm_bahdanau.py

@author: xiaolu

@time  : 2019-07-24

"""
"""

@file  : lstm_luong注意力.py

@author: xiaolu

@time  : 2019-07-23

"""
from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import sklearn
import numpy as np
from sklearn.metrics import classification_report


trainset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
print(trainset.target_names)
print(len(trainset.data))
print(len(trainset.target))

ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(trainset.data,
                                                                               trainset.target,
                                                                               ONEHOT, test_size=0.2)

concat = ' '.join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
# 构建词典
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print('vocab from size: %d' % vocabulary_size)
print('Most common words', count[4:10])
print('Sample data', data[:10], [rev_dictionary[i] for i in data[:10]])

GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']


class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, maxlen,
                 grad_clip=5.0, kernel_sizes=[3, 3, 3]):
        '''
        :param size_layer: 每步输出维度
        :param num_layers: rnn的层数
        :param embedded_size: 词嵌入维度
        :param dict_size: 字典大小
        :param dimension_output: 类别数
        :param maxlen: 最大长度
        :param grad_clip: 梯度裁剪
        :param kernel_sizes: 卷积核行数   列数不需要指定 必须是embedding_size
        '''
        n_filters = [25 * k for k in kernel_sizes]   # [75, 75, 75]

        def cells(reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size_layer, initializer=tf.orthogonal_initializer(), reuse=reuse)

        def add_highway(x, i):
            size = sum(n_filters)
            reshaped = tf.reshape(x, [-1, size])    # shape=(batch_size, n_filters)
            H = tf.layers.dense(reshaped, size, tf.nn.relu, name='activation' + str(i))
            T = tf.layers.dense(reshaped, size, tf.sigmoid, name='transform_gate' + str(i))
            C = tf.subtract(1.0, T)    # 1.0 - T  subtract是减法操作 并且支持广播机制
            highway_out = tf.add(tf.multiply(H, T), tf.multiply(reshaped, C))
            print("highway_out:", highway_out.shape)
            return tf.reshape(highway_out, [-1, 1, size])

        # 1. define input
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        # 因为我们后面要卷积　所以这里需要reshape一下
        encoder_embedded = tf.reshape(encoder_embedded, [-1, maxlen, embedded_size])

        parallels = []
        for i, (n_filter, kernel_size) in enumerate(zip(n_filters, kernel_sizes)):
            conv_out = tf.layers.conv1d(inputs=encoder_embedded,
                                        filters=n_filter,   # 75  也就是每次使用75个卷积和及你想嗯卷积
                                        kernel_size=kernel_size,   # 3
                                        activation=tf.tanh,
                                        name='conv1d' + str(i)
                                        )

            pool_out = tf.layers.max_pooling1d(inputs=conv_out,
                                               pool_size=conv_out.get_shape().as_list()[1],
                                               strides=1
                                               )     # 直接将一个竖条转为一个标量
            parallels.append(tf.reshape(pool_out, [-1, n_filter]))

        pointer = tf.concat(parallels, 1)    # shape = (batch_size, n_filter)
        print("conv_shape:", pointer.shape)

        for i in range(2):
            pointer = add_highway(pointer, i)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        outputs, _ = tf.nn.dynamic_rnn(rnn_cells, pointer, dtype=tf.float32)

        W = tf.get_variable('w', shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output), initializer=tf.zeros_initializer())

        self.logits = tf.matmul(outputs[:, -1], W) + b

        # 计算损失
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        # 梯度裁剪
        params = tf.trainable_variables()
        gradients = tf.gradients(self.cost, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)

        self.optimizer = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


tf.reset_default_graph()
sess = tf.Session()


model = Model(
    size_layer=size_layer,
    num_layers=num_layers,
    embedded_size=embedded_size,
    dict_size=vocabulary_size+4,
    dimension_output=dimension_output,
    maxlen=maxlen,
)
sess.run(tf.global_variables_initializer())

# 开始训练
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print('break epoch:%d\n' % (EPOCH))
        break

    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i:i + batch_size], dictionary, maxlen)
        acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                feed_dict={model.X: batch_x, model.Y: train_onehot[i:i + batch_size]})
        train_loss += loss
        train_acc += acc

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i:i + batch_size], dictionary, maxlen)
        acc, loss = sess.run([model.accuracy, model.cost],
                             feed_dict={model.X: batch_x, model.Y: test_onehot[i:i + batch_size]})
        test_loss += loss
        test_acc += acc

    train_loss /= (len(train_X) // batch_size)
    train_acc /= (len(train_X) // batch_size)
    test_loss /= (len(test_X) // batch_size)
    test_acc /= (len(test_X) // batch_size)

    if test_acc > CURRENT_ACC:
        print('epoch: %d, pass acc: %f, current acc: %f' % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print('time taken:', time.time() - lasttime)
    print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n' % (EPOCH, train_loss,
                                                                                               train_acc, test_loss,
                                                                                               test_acc))
    EPOCH += 1

logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names))
