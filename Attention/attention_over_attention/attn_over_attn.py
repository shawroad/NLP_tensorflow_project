"""

@file  : attn_over_attn.py

@author: xiaolu

@time  : 2019-07-17

"""
import tensorflow as tf


class Model(object):
    def __init__(self,
                 size_layer,
                 num_layers,
                 embedded_size,
                 dict_size,
                 dimension_output,
                 learning_rate,
                 maxlen
                 ):
        '''
        :param size_layer: 每步输出维度
        :param num_layers: 有多少层
        :param embedded_size: 词嵌入的维度
        :param dict_size: 词表的大小
        :param dimension_output: 输出维度
        :param learning_rate: 学习率
        :param maxlen: 填充后的长度
        :return:
        '''
        def cells(reuse=False):
            return tf.nn.rnn_cell.GRUCell(size_layer, reuse=reuse)

        # 1. 定义输入
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        # 2. 词嵌入　　进行两次词嵌入
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embeddings_query = tf.Variable(tf.random_uniform([dict_size, embedded_size], -5, 5))

        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        encoder_embedded_query = tf.nn.embedding_lookup(encoder_embeddings_query, self.X)

        # 3. 两个词嵌入分别进入两个多层搞得rnn中
        with tf.variable_scope('document', initializer=tf.orthogonal_initializer()):
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

        with tf.variable_scope('query', initializer=tf.orthogonal_initializer()):
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs_query, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded_query, dtype=tf.float32)

        M = tf.multiply(outputs, outputs_query)
        alpha = tf.nn.softmax(M, 1)
        print(alpha.shape)

        beta = tf.nn.softmax(M, 2)
        print(beta.shape)

        query_importance = tf.expand_dims(tf.reduce_sum(beta, 1), -1)
        print(query_importance.shape)

        s = tf.squeeze(tf.matmul(alpha, query_importance), 2)

        W = tf.get_variable('w', shape=(maxlen, dimension_output), initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output,), initializer=tf.zeros_initializer())

        self.logits = tf.matmul(s, W) + b

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
