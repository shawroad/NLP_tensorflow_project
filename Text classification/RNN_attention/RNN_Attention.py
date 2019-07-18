"""

@file  : CNN_RNN.py

@author: xiaolu

@time  : 2019-07-18

"""
import tensorflow as tf

class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, num_classes, learning_rate):
        '''
        :param size_layer: 每层输出
        :param num_layers: 多少层
        :param embedded_size: 词嵌入维度
        :param dict_size: 词表大小
        :param num_classes: 类别
        :param learning_rate: 学习率
        '''
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, num_classes])

        # 词嵌入
        self.encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.X)

        # define net
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(size_layer)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
        drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob=0.5)
        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, encoder_embedded, dtype=tf.float32)

        # define attention
        attention_size = 64
        attention_w = tf.get_variable('attention_v', [attention_size], tf.float32)
        query = tf.layers.dense(tf.expand_dims(self.last_state[-1].h, 1), attention_size)  # 将最后的隐态dense 为attention_size
        # outputs: shape(batch, time_series, layer_size)
        keys = tf.layers.dense(self.outputs, attention_size)    # 把每层的输出dense成attention_size
        align = tf.reduce_sum(attention_w * tf.tanh(keys + query), [2])
        align = tf.nn.softmax(align)
        self.outputs = tf.squeeze(tf.matmul(tf.transpose(self.outputs, [0, 2, 1]),
                                            tf.expand_dims(align, 2)), 2)

        self.rnn_W = tf.Variable(tf.random_normal((size_layer, num_classes)))
        self.rnn_B = tf.Variable(tf.random_normal(([num_classes])))
        self.logits = tf.matmul(self.outputs, self.rnn_W) + self.rnn_B

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))