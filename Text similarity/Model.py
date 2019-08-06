"""

@file  : Model.py

@author: xiaolu

@time  : 2019-08-05

"""
import tensorflow as tf
import numpy as np


def position_encoding(inputs):
    T = tf.shape(inputs)[1]
    repr_dim = inputs.get_shape()[-1].value
    pos = tf.reshape(tf.range(0.0, tf.to_float(T), dtype=tf.float32), [-1, 1])
    i = np.arange(0, repr_dim, 2, np.float32)
    denom = np.reshape(np.power(10000.0, i / repr_dim), [1, -1])
    enc = tf.expand_dims(tf.concat([tf.sin(pos / denom), tf.cos(pos / denom)], 1), 0)
    return tf.tile(enc, [tf.shape(inputs)[0], 1, 1])


def layer_norm(inputs, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))
    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())
    return gamma * normalized + beta


def self_attention(inputs, is_training, num_units, num_heads=8, activation=None):
    T_q = T_k = tf.shape(inputs)[1]
    Q_K_V = tf.layers.dense(inputs, 3 * num_units, activation)
    Q, K, V = tf.split(Q_K_V, 3, -1)
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), 0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), 0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), 0)
    align = tf.matmul(Q_, K_, transpose_b=True)
    align *= tf.rsqrt(tf.to_float(K_.get_shape()[-1].value))
    paddings = tf.fill(tf.shape(align), float('-inf'))
    lower_tri = tf.ones([T_q, T_k])
    lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
    masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
    align = tf.where(tf.equal(masks, 0), paddings, align)
    align = tf.nn.softmax(align)
    align = tf.layers.dropout(align, 0.1, training=is_training)
    x = tf.matmul(align, V_)
    x = tf.concat(tf.split(x, num_heads, axis=0), 2)
    x += inputs
    x = layer_norm(x)
    return x


def ffn(inputs, hidden_dim, activation=tf.nn.relu):
    x = tf.layers.conv1d(inputs, 4 * hidden_dim, 1, activation=activation)
    x = tf.layers.conv1d(x, hidden_dim, 1, activation=None)
    x += inputs
    x = layer_norm(x)
    return x


class Model:
    def __init__(self, size_layer, num_layers, embedded_size,
                 dict_size, learning_rate, kernel_size=5):
        def cnn(x, scope):
            x += position_encoding(x)
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                for n in range(num_layers):
                    with tf.variable_scope('attn_%d' % n, reuse=tf.AUTO_REUSE):
                        x = self_attention(x, True, size_layer)
                    with tf.variable_scope('ffn_%d' % n, reuse=tf.AUTO_REUSE):
                        x = ffn(x, size_layer)

                with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
                    return tf.layers.dense(x, size_layer)[:, -1]

        self.X_left = tf.placeholder(tf.int32, [None, None])
        self.X_right = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None])
        self.batch_size = tf.shape(self.X_left)[0]
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        embedded_left = tf.nn.embedding_lookup(encoder_embeddings, self.X_left)
        embedded_right = tf.nn.embedding_lookup(encoder_embeddings, self.X_right)

        def contrastive_loss(y, d):
            tmp = y * tf.square(d)
            tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
            return tf.reduce_sum(tmp + tmp2) / tf.cast(self.batch_size, tf.float32) / 2

        self.output_left = cnn(embedded_left, 'left')
        self.output_right = cnn(embedded_right, 'right')
        print(self.output_left, self.output_right)
        self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.output_left, self.output_right)),
                                              1, keep_dims=True))
        self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.output_left),
                                                                           1, keep_dims=True)),
                                                     tf.sqrt(tf.reduce_sum(tf.square(self.output_right),
                                                                           1, keep_dims=True))))
        self.distance = tf.reshape(self.distance, [-1])
        self.cost = contrastive_loss(self.Y, self.distance)

        self.temp_sim = tf.subtract(tf.ones_like(self.distance),
                                    tf.rint(self.distance))
        correct_predictions = tf.equal(self.temp_sim, self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
