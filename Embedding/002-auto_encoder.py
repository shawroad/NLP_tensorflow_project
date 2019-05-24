"""

@file   : 002-auto_encoder.py

@author : xiaolu

@time1  : 2019-05-24

"""
import sklearn.datasets
from sklearn.model_selection import train_test_split
import re
import tensorflow as tf
import numpy as np
import collections

# 思路: 自编码器这是一种无监督的方式训练句向量
# 1. 将句子塞进多层的LSTM网络中，最后加一个全连接层 权重的格式[输出的维度, 要把它压缩成多长(就是咱们要得到的句向量)]
# 2. 1中全连接层之后再连接一个全连接层，这两个全连接层的维度刚好是相反的
# 3. 从2中输出来向量，我们将形状reshape 然后输入同样规格的LSMT解码
# 4. 最后定义损失， 输出的词和我们反向输出进行对比 计算交叉损失熵

def clearstring(string):
    # 对文本进行清洗
    string = re.sub('[^A-Za-z ]+', '', string)
    string = string.split('\n')
    string = [y.strip() for y in filter(None, string)]
    string = (' '.join(string)).lower()
    return ' '.join([y.strip() for y in string.split()])


def separate_dataset(trainset):
    # 数据集切分  特征及标签
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


def build_dataset(words, n_words):
    # 构造词典，并且填充不够的语料
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words))
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


def str_idx(corpus, dic, maxlen, UNK=3):
    # 来一批数据 我们将其转成等长用id表示的文本
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            try:
                X[i, -1 - no] = dic[k]
            except Exception as e:
                X[i, -1 - no] = UNK
    return X

# 加载数据
dataset = sklearn.datasets.load_files(container_path = 'data', encoding = 'UTF-8')
# 进行数据的切分  分的是特征和标签
dataset.data, dataset.target = separate_dataset(dataset)
# 训练集和测试集的切分
_, dataset.data, _, dataset.target = train_test_split(dataset.data, dataset.target, test_size=0.03)
print(len(dataset.data))   # 训练集占的数量  320


# 对训练集切分 获得词表 以及id的映射
concat = ' '.join(dataset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
print('词典的大小:', vocabulary_size)
print('出现次数最多的六个词:', count[4:10])   # 因为前四个词使我们的填充标志
print('查看一些数据及对应的id:', data[:10], [rev_dictionary[i] for i in data[:10]])


GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']


# 模型的定义
class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,seq_len):
        # size_layer: 总共有多少层LSTM
        # num_layers: 一层有多少个LSTM单元 也就是循环几次
        # embedded_size: 是词嵌入的维度
        # dict_size: 词典的大小  在进行词嵌入的时候需要这个数值
        # dimension_output: LSTM出来后，我们有连一个全连接层，这个是全连接层的输出
        # learning_rate: 学习率
        # seq_len: 每一层LSTM的输出长度

        # 1.定义一个基本的LSTM单元
        def cells(size, reuse=False):
            return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse)

        # 2.定义占位符
        self.X = tf.placeholder(tf.int32, [None, None])

        # 3.词嵌入初始化 及词嵌入
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], 0, 1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)

        # 4.建立多层的LSTM  并指定一层要循环几次
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells(size_layer) for _ in range(num_layers)])
        outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

        # 5.输出
        outputs = tf.reshape(outputs, [-1, seq_len*size_layer])

        # 6.然后经过一个全连接层  初始化权重和偏置  再经过一个tanh激活函数
        W = tf.get_variable('w', shape=(size_layer*seq_len, dimension_output), initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output, ), initializer=tf.zeros_initializer())
        self.logits = tf.nn.tanh(tf.matmul(outputs, W) + b)
        # 上面是编码的过程

        # 开始解码
        # 7.这里根据6的全连接 在全连接  注意这里的输入和输出和6中的刚好相反
        W_decoder = tf.get_variable('w_decoder', shape=(dimension_output, size_layer*seq_len), initializer=tf.orthogonal_initializer())
        b_decoder = tf.get_variable('b_decoder', shape=(size_layer*seq_len), initializer=tf.zeros_initializer())
        logits_decoder = tf.nn.tanh(tf.matmul(self.logits, W_decoder) + b_decoder)

        # 8. 全连接完了后 将其形状reshape成 (-1, seq_len, size_layer)  搞成这样 我们是为了重新让其进如反向的LSTM
        logits_decoder = tf.reshape(logits_decoder, [-1, seq_len, size_layer])
        with tf.variable_scope("decoder"):
            rnn_cells_decoder = tf.nn.rnn_cell.MultiRNNCell([cells(embedded_size) for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells_decoder, logits_decoder, dtype=tf.float32)

        # 最后将输出大小整成词表的大小
        decoder_dense = tf.layers.dense(outputs, dict_size)

        # 获取语料的one_hot码  然后和decoder_dense进行计算交叉损失熵
        onehot_X = tf.one_hot(self.X, dict_size, axis=2, dtype=tf.float32)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_X, logits=decoder_dense)
        self.cost = tf.reduce_mean(loss)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.cost)


size_layer = 256
num_layers = 2
embedded_size = 128
dimension_output = 100
learning_rate = 1e-2
maxlen = 50
batch_size = 32
epoch = 10

tf.reset_default_graph()
sess = tf.InteractiveSession()
model = Model(size_layer, num_layers, embedded_size, vocabulary_size+4, dimension_output,learning_rate,maxlen)
sess.run(tf.global_variables_initializer())


X = str_idx(dataset.data, dictionary, maxlen)
Y = dataset.target


for i in range(epoch):
    total_loss = 0
    for k in range(0, (X.shape[0]//batch_size)*batch_size, batch_size):
        loss, _ = sess.run([model.cost, model.optimizer], feed_dict={model.X:X[k:k+batch_size,:]})
        total_loss += loss
    total_loss /= (X.shape[0]//batch_size)
    print('epoch %d, avg loss %f'% (i+1, total_loss))

logits_test = sess.run(tf.nn.sigmoid(model.logits), feed_dict={model.X: X})
print(logits_test.shape)   # 将测试集塞进去 编码最后的输出 (数据, dimension_size)

print(logits_test)   # 输出对应的句子向量


