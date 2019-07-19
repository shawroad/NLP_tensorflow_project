"""

@file  : LSTM_Attention_net.py

@author: xiaolu

@time  : 2019-07-19

"""
"""

@file  : BiRnn_attention.py

@author: xiaolu

@time  : 2019-07-19

"""
import tensorflow as tf
from MyAttention import attention


class Model:
    # 构建带注意力的双向rnn
    def __init__(self, sequence_length, num_classes, embedding_size, vocab_size,
                 hidden_size, rnn_layer_size, attention_size, l2_reg_lambda, learning_rate):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_data')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_label')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1, name='W'))
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            inputs = tf.transpose(self.embedded_chars, [1, 0, 2])
            inputs = tf.reshape(inputs, [-1, embedding_size])
            inputs = tf.split(inputs, sequence_length, 0)  # 双向rnn中需要输入的是list

        with tf.name_scope('build_rnn'):
            with tf.name_scope('fw_rnn'):
                fw_cell_list = [tf.contrib.rnn.LSTMCell(hidden_size) for _ in range(rnn_layer_size)]
                fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(fw_cell_list),
                                                          output_keep_prob=(1 - self.dropout_rate))
            with tf.name_scope("bw_rnn"):
                bw_cell_list = [tf.contrib.rnn.LSTMCell(hidden_size) for _ in range(rnn_layer_size)]
                bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(bw_cell_list),
                                                          output_keep_prob=(1 - self.dropout_rate))
            with tf.name_scope('bi_rnn'):
                self.outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell_m, bw_cell_m, inputs,
                                                                                                    dtype=tf.float32)

        with tf.name_scope('outputs'):
            self.attention_output1, self.alphas1 = attention(self.outputs, attention_size, return_alphas=True)
            w_p = tf.get_variable('w', [hidden_size * 2, num_classes], dtype=tf.float32)
            b_p = tf.get_variable('b', [num_classes], dtype=tf.float32)
            self.scores = tf.nn.xw_plus_b(self.attention_output1, w_p, b_p, name='scores')

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,
                                                                            labels=self.input_y,
                                                                            name="cross_entropy")
            vars = tf.trainable_variables()
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_reg_lambda

            self.loss = tf.reduce_mean(self.cross_entropy + self.l2_losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')


