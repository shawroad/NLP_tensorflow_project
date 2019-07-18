"""

@file  : CNN_RNN.py

@author: xiaolu

@time  : 2019-07-18

"""
import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, vocab_size, embedding_size, sequence_length, dimension_output,
                 learning_rate, filter_sizes, pooling_size, out_dimension, num_layers):
        # 　size_layer = 64
        # num_layers = 2
        # sequence_length= 20
        # embedded_size = 128
        # dict_size = len(vocab2id)
        # num_classes = 2
        # learning_rate = 0.01
        # filter_sizes = [3, 3, 3]
        # pooling_size = 2
        # model = Model(vocab_size=dict_size, embedding_size=embedded_size, sequence_length=sequence_length,
        # dimension_output=num_classes, learning_rate=learning_rate, filter_sizes=filter_sizes,
        # pooling_size=pooling_size, out_dimension=size_layer, num_layers=num_layers
        # )

        self.X = tf.placeholder(tf.int32, shape=[None, None])
        self.Y = tf.placeholder(tf.float32, shape=[None, dimension_output])

        # embedding
        self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.X)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # conv
        pooled_outputs = []
        reduce_size = int(np.ceil((sequence_length) * 1.0 / pooling_size))
        for i in filter_sizes:
            w = tf.Variable(tf.truncated_normal([i, embedding_size, 1, out_dimension], stddev=0.1))
            b = tf.Variable(tf.truncated_normal([out_dimension], stddev=0.01))
            conv = tf.nn.relu(tf.nn.conv2d(self.embedded_chars_expanded, w, strides=[1, 1, 1, 1], padding="VALID") + b)
            pooled = tf.nn.max_pool(conv, ksize=[1, pooling_size, 1, 1], strides=[1, pooling_size, 1, 1],padding='VALID')
            pooled = tf.reshape(pooled, [-1, reduce_size - 1, out_dimension])
            pooled_outputs.append(pooled)
        h_pool = tf.concat(pooled_outputs, 2)

        # lstm
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(out_dimension)

        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob=0.5)

        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, h_pool, dtype=tf.float32)

        self.rnn_W = tf.Variable(tf.random_normal((out_dimension, dimension_output)))
        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))
        self.logits = tf.matmul(self.outputs[:, -1], self.rnn_W) + self.rnn_B

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

