"""

@file   : 001-CBOW之softmax实现.py

@author : xiaolu

@time1  : 2019-05-24

"""
from utils import *
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np

# 加载数据
trainset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')
trainset.data, trainset.target = separate_dataset(trainset, 1)

# 我们原本文本是多行，这里我们将所有文本连起来当做语料库 进行词向量的训练
texts = ' '.join(trainset.data)  # 将所有语料连起来
words = texts.split()  # 又进行分词
word2freq = Counter(words)  # 统计词频
print("语料中总的单词数:", len(words))

_words = set(words)  # 将词进行去重  构建词表
word2idx = {c: i for i, c in enumerate(_words)}
idx2word = {i: c for i, c in enumerate(_words)}
vocab_size = len(idx2word)   # 词表的大小
indexed = [word2idx[w] for w in words]  # 将语料中所有词转为对应的id
print("词表的大小(也就是语料去重后单词数):", vocab_size)


class CBOW:
    def __init__(self, sample_size, vocab_size, embedded_size, window_size=3):
        # window_size:取中心词的上文中window_size个词，去中心词下文中window_size个词
        # vocab_size:词表的大小
        # enbedded_size:词嵌入的维度
        # sample_size: 批量的大小

        # 1. 定义占位符
        self.X = tf.placeholder(tf.int32, shape=[None, 2*window_size])
        self.Y = tf.placeholder(tf.int32, shape=[None, 1])

        # 2. 初始化输入的维度
        self.embedding = tf.Variable(tf.truncated_normal([vocab_size, embedded_size], stddev=1.0 / np.sqrt(embedded_size)))
        self.bias = tf.Variable(tf.zeros([vocab_size]))
        embedded = tf.nn.embedding_lookup(self.embedding, self.X)
        embedded = tf.reduce_mean(embedded, axis=1)

        # 定义损失  这里最后的softmax采用的是加速的softmax  tensorflow封装在sampled_softmax_loss中
        self.cost = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.embedding,
            biases=self.bias,
            labels=self.Y,
            inputs=embedded,
            num_sampled=sample_size,
            num_classes=vocab_size
        ))
        # 定义优化器
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

        # 输入一个词
        self.valid_dataset = tf.placeholder(tf.int32, shape=[None])  # 输入一些词，我们每训练一轮，让其找这些词的相似词

        # 这里是获取所有嵌入的词向量
        norm = tf.sqrt(tf.reduce_mean(tf.square(self.embedding), 1, keep_dims=True))
        normalized_embeddings = self.embedding / norm     # 获取的是所有词的词向量，这里进行了归一化

        # 获取输入词的词向量
        self.valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, self.valid_dataset)

        # 获取给定词对应的相似词
        self.similarity = tf.matmul(self.valid_embeddings, normalized_embeddings, transpose_b=True)


# 一些参数定义
batch_size = 128
embedded_size = 128
window_size = 3
epoch = 10
valid_size = 10   # 随机选取十个词，等会找这十个词最相似的词
nearest_neighbors = 8    # 找上面词最相似的八个词


tf.reset_default_graph()
sess = tf.InteractiveSession()
model = CBOW(batch_size, vocab_size, embedded_size)
sess.run(tf.global_variables_initializer())


def get_x(words, idx):
    left = idx - window_size
    right = idx + window_size
    return words[left: idx] + words[idx+1: right+1]


def make_xy(int_words):
    # int_words:是将语料转化为对应数字的列表
    x, y = [], []
    for i in range(window_size, len(int_words)-window_size):
        inputs = get_x(int_words, i)    # 给定一个窗口，然后构造两组数据[[上文词, 目标词], [下文词, 目标词]]
        x.append(inputs)   # 构造的输入
        y.append(int_words[i])   # 对应的输出
    return np.array(x), np.array(y)


X, Y = make_xy(indexed)


for i in range(epoch):
    total_cost = 0
    for k in range(0, (X.shape[0] // batch_size) * batch_size, batch_size):
        batch_x = X[k: k+batch_size]
        batch_y = Y[k: k+batch_size, np.newaxis]
        cost, _ = sess.run([model.cost, model.optimizer], feed_dict={model.X: batch_x, model.Y: batch_y})
        total_cost += cost   # 累加损失
    total_cost /= (X.shape[0] // batch_size)   # 平均每批的损失
    print('epoch %d, avg loss %f' % (i+1, total_cost))


    # 每次随机选取十个词，找对应最接近的词
    random_valid_size = np.random.choice(indexed, valid_size)
    similarity = sess.run(model.similarity, feed_dict={model.valid_dataset: random_valid_size})
    # vector = sess.run(model.valid_embeddings, feed_dict={model.valid_dataset: random_valid_size})   # 这里获取的是每个词的词向量
    for no, i in enumerate(random_valid_size):
        # print(vector)   # 打印对应词的词向量
        valid_word = idx2word[i]
        nearest = (-similarity[no, :]).argsort()[1:nearest_neighbors + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(nearest_neighbors):
            close_word = idx2word[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

