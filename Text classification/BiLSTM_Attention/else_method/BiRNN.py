"""

@file  : BiRNN.py

@author: xiaolu

@time  : 2019-07-20

"""
import tensorflow as tf
import numpy as np

class BiRNN:
    def __init__(self, embedding_size, rnn_size, layer_size, vocab_size, attn_size, sequence_length,
                 n_classes, l2_reg_lambda, learning_rate):
        '''
        :param embedding_size: word embedding dimension
        :param rnn_size: hidden state dimension
        :param layer_size: number of rnn layers
        :param vocab_size: vocabulary size
        :param attn_size: attention layer dimension
        :param sequence_length: max sequence length
        :param n_classes: number of target labels
        :param grad_clip: gradient clipping threshold
        :param learning_rate: initial learning rate
        '''
        self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
        self.input_data = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_data')
        self.targets = tf.placeholder(tf.float32, shape=[None, n_classes], name='targets')

        with tf.name_scope('embedding'):
            embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1), name='embedding')
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # define fw_rnn_cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.output_keep_prob)

        # define bw_rnn_cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=self.output_keep_prob)


        # self.input_data shape: (batch_size , sequence_length)
        # inputs shape : (batch_size , sequence_length , rnn_size)
        inputs = tf.transpose(inputs, [1, 0, 2])
        inputs = tf.reshape(inputs, [-1, rnn_size])
        inputs = tf.split(inputs, sequence_length, 0)

        # define birnn
        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs,
                                                                    dtype=tf.float32)
        # 这里的outputs默认是将正反方向的结果拼接到一起　
        # define attention
        attention_size = attn_size
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2*rnn_size, attention_size], stddev=0.1),
                                      name='attention_w')

            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')

            u_list = []
            for t in range(sequence_length):
                # 将每步输出dense到attention_size维度
                u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in range(sequence_length):
                # 每步dense出的attention_size 矩阵乘 随机化的attention向量, 将其压缩压缩为一维度
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # attn_z 的shape为: batch_size, sequence_length, 1


            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            print(attn_zconcat.shape)   # shape=(?, 20) 因为我们序列长度为20
            self.alpha = tf.nn.softmax(attn_zconcat)

            # transform to sequence length*batch_size * 1
            alpha_trans = tf.reshape(tf.transpose(self.alpha, [1, 0]), [sequence_length, -1, 1])
            self.final_output = tf.reduce_sum(outputs * alpha_trans, 0)

            # print(self.final_output.shape)  # (?, 128)

        with tf.name_scope('output'), tf.variable_scope('output'):
            # outputs shape: (sequence_length, batch_size, 2*rnn_size)
            fc_w = tf.Variable(tf.truncated_normal([2*rnn_size, n_classes], stddev=0.1, name='fc_w'))
            fc_b = tf.Variable(tf.zeros([n_classes]), name='fc_b')

            # 用于分类任务, outputs取最终一个时刻的输出
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                            labels=self.targets,
                                                                            name="cross_entropy")
            vars = tf.trainable_variables()
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_reg_lambda

            self.loss = tf.reduce_mean(self.cross_entropy + self.l2_losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            self.predictions = tf.argmax(self.logits, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')


