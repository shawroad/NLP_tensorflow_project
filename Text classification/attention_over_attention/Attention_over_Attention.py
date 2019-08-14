"""

@file  : Attention_over_Attention.py

@author: xiaolu

@time  : 2019-07-18

"""
import tensorflow as tf


class Model:
    def __init__(self, size_layer, num_layers, embedded_size,
                 dict_size, dimension_output, learning_rate, maxlen):
        '''
        :param size_layer: 每步的输出维度
        :param num_layers: 多少层GRU
        :param embedded_size: 词嵌入的维度
        :param dict_size: 词典的大小
        :param dimension_output: 类别的个数
        :param learning_rate: 学习率
        :param maxlen: 最大长度
        '''
        def cells(reuse=False):
            return tf.nn.rnn_cell.GRUCell(size_layer, reuse=reuse)

        self.input_x = tf.placeholder(tf.int32, [None, None])
        self.input_y = tf.placeholder(tf.float32, [None, dimension_output])

        # 1 进行两次词嵌入的过程
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embeddings_query = tf.Variable(tf.random_uniform([dict_size, embedded_size], -5, 5))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.input_x)
        encoder_embedded_query = tf.nn.embedding_lookup(encoder_embeddings_query, self.input_x)

        # 2 定义几层GRU  针对第一次词嵌入
        with tf.variable_scope('document', initializer=tf.orthogonal_initializer()):
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)

        # 3 定义几层GRU 针对第二次词嵌入
        with tf.variable_scope('query', initializer=tf.orthogonal_initializer()):
            rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
            outputs_query, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded_query, dtype=tf.float32)

        # (?, ?, 64)
        # (?, ?, 64)
        # (?, 64, 1)
        # (?, ?)
        M = tf.multiply(outputs, outputs_query)  # 两个GRU 对应的输出相乘
        alpha = tf.nn.softmax(M, 1)
        print(alpha.shape)

        beta = tf.nn.softmax(M, 2)
        print(beta.shape)

        query_importance = tf.expand_dims(tf.reduce_sum(beta, 1), -1)
        print(query_importance.shape)

        s = tf.squeeze(tf.matmul(alpha, query_importance), 2)
        print(s.shape)

        W = tf.get_variable('w', shape=(maxlen, dimension_output), initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output), initializer=tf.zeros_initializer())

        self.logits = tf.nn.xw_plus_b(s, W, b)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
