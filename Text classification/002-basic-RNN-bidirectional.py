"""

@file   : 002-basic-RNN-bidirectional.py

@author : xiaolu

@time1  : 2019-05-24

"""
from utils import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import sklearn
import numpy as np
from sklearn.metrics import classification_report


# 加载数据
trainset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')
trainset.data, trainset.target = separate_dataset(trainset, 1)

# 将标签整成one_hot编码
one_hot = np.zeros((len(trainset.data), len(trainset.target_names)))
one_hot[np.arange(len(trainset.data)), trainset.target] = 1.0


# 切分数据集
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(trainset.data, trainset.target, one_hot, test_size=0.2, random_state=42)

concat = ' '.join(trainset.data).split()  # 将所有句子连一块 然后进行分词
vocabulary_size = len(list(set(concat)))  # 得到词表的大小
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
# print("词表的大小:", vocabulary_size)
# print("前10个常用词:", count[4:10])

GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']


# 定义模型
class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate):

        def cells(size, reuse=False):
            return tf.nn.rnn_cell.BasicRNNCell(size, reuse=reuse)

        # 1.占位符
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        # 2.词嵌入
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))  # 词嵌入的输入
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        # 3.定义双向的RNN
        for n in range(num_layers):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells(size_layer // 2),
                cell_bw=cells(size_layer // 2),
                inputs=encoder_embedded,
                dtype=tf.float32,
                scope='bidirectional_rnn_%d' % (n)
            )
            encoder_embedded = tf.concat((out_fw, out_bw), 2)  # 将前向和反向的输出合并

        # 4.初始化一些参数
        W = tf.get_variable('w', shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output, ), initializer=tf.zeros_initializer())

        # 5. 最后一层输出然后连一个全连接层
        self.logits = tf.matmul(encoder_embedded[:, -1], W) + b   # 之所以不要最后一行 是因为-1那个是填充的

        # 6. 定义损失
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        # 7.优化器
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # 8. 看有那些预测正确
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        # 正确率
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 参数
size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 50
batch_size = 128


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(size_layer, num_layers, embedded_size, vocabulary_size+4, dimension_output, learning_rate)
sess.run(tf.global_variables_initializer())

# 训练
EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:  # 超过5个epoch停止  英文用的是cpu只是简单让其训练5个epoch
        print('break epoch:%d\n'%(EPOCH))
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

    print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n' % (EPOCH, train_loss,
                                                                                               train_acc,
                                                                                               test_loss,
                                                                                               test_acc))
    EPOCH += 1


logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names))
