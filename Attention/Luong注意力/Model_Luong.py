"""

@file  : Model_Luong.py

@author: xiaolu

@time  : 2019-07-17

"""
import tensorflow as tf
import numpy as np


class Attention:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.dense_layer = tf.layers.Dense(hidden_size)
        self.v = tf.random_normal([hidden_size], mean=0, stddev=1 / np.sqrt(hidden_size))

    def score(self, hidden_tensor, encoder_outputs):
        energy = tf.nn.tanh(self.dense_layer(encoder_outputs))
        energy = tf.transpose(energy, [0, 2, 1])
        batch_size = tf.shape(encoder_outputs)[0]
        v = tf.expand_dims(tf.tile(tf.expand_dims(self.v, 0), [batch_size, 1]), 1)
        energy = tf.matmul(v, energy)
        return tf.squeeze(energy, 1)

    def __call__(self, hidden, encoder_outputs):
        seq_len = tf.shape(encoder_outputs)[1]
        batch_size = tf.shape(encoder_outputs)[0]
        H = tf.tile(tf.expand_dims(hidden, 1), [1, seq_len, 1])
        attn_energies = self.score(H, encoder_outputs)
        return tf.expand_dims(tf.nn.softmax(attn_energies), 1)


class Luong(tf.contrib.rnn.RNNCell):
    def __init__(self, hidden_size, output_size, encoder_outputs):
        self.hidden_size = hidden_size
        self.batch_size = tf.shape(encoder_outputs)[0]
        self.gru = tf.contrib.rnn.GRUCell(hidden_size)
        self.attention = Attention(hidden_size)
        self.out = tf.layers.Dense(output_size)
        self.encoder_outputs = encoder_outputs
        self.reset_state()

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size

    def reset_state(self):
        self.context = tf.zeros(shape=(self.batch_size, self.hidden_size))

    def __call__(self, inputs, state, scope=None):
        rnn_input = tf.concat([inputs, self.context], 1)
        output, hidden = self.gru(rnn_input, state)
        attn_weights = self.attention(output, self.encoder_outputs)
        self.context = tf.matmul(attn_weights, self.encoder_outputs)[:, 0, :]
        output = tf.concat([output, self.context], 1)
        output = tf.nn.softmax(self.out(output))
        return output, hidden


class Model:
    def __init__(self, size_layer, embedded_size, dict_size, dimension_output, learning_rate):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])

        self.encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, 1))
        encoder_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, self.X)

        self.luong_cell = Luong(size_layer, size_layer, encoder_embedded)

        outputs, last_states = tf.nn.dynamic_rnn(self.luong_cell, encoder_embedded, dtype=tf.float32)

        W = tf.get_variable('w', shape=(size_layer, dimension_output), initializer=tf.orthogonal_initializer())
        b = tf.get_variable('b', shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs[:, -1], W) + b

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))