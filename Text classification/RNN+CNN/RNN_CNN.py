"""

@file  : RNN_CNN.py

@author: xiaolu

@time  : 2019-07-18

"""
import tensorflow as tf


class Model:
    def __init__(self, sequence_length, num_classes, vocab_size, batch_size, embedding_size,
                 hidden_size, filter_sizes, num_filters, rnn_layer_size, l2_reg_lambda, learning_rate):

        self.batch_size = batch_size
        # 1. define placeholder
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_data')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_label')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')

        # 2. define embedding
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # inputs shape : (batch_size , sequence_length , rnn_size)
            # bidirection rnn 的inputs shape 要求是(sequence_length, batch_size, embedding_size)
            # 因此这里需要对inputs做一些变换
            # 经过transpose的转换已经将shape变为(sequence_length, batch_size, embedding_size)
            # 只是双向rnn接受的输入必须是一个list,因此还需要后续两个步骤的变换
            inputs = tf.transpose(self.embedded_chars, [1, 0, 2])
            # 转换成(batch_size * sequence_length, embedding_size)
            inputs = tf.reshape(inputs, [-1, embedding_size])
            # 转换成list,里面的每个元素是(batch_size, embedding_size)
            inputs = tf.split(inputs, sequence_length, 0)  # 沿着轴0进行切割, 切成一个样本一个样本

        # 3. define bilstm
        with tf.name_scope('build_rnn'):
            # fw
            with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
                fw_cell_list = [tf.contrib.rnn.LSTMCell(hidden_size) for _ in range(rnn_layer_size)]
                fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(fw_cell_list),
                                                          output_keep_prob=(1-self.dropout_rate))
            # bw
            with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
                bw_cell_list = [tf.contrib.rnn.LSTMCell(hidden_size) for _ in range(rnn_layer_size)]
                bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(bw_cell_list),
                                                          output_keep_prob=(1-self.dropout_rate))

            # combine_direction
            with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
                outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell_m, bw_cell_m, inputs,
                                                                        dtype=tf.float32)

            out = tf.layers.dense(tf.concat(outputs, -1),
                                  sequence_length * embedding_size,
                                  activation=tf.nn.relu)   # bilstm最后输出接入一个dense 输出维度为序列长度＊embedding

            final_output = tf.reshape(out, [-1, sequence_length, embedding_size])  # 调整送进cnn

        # define conv
        with tf.name_scope('build_cnn'):
            cnn_input = tf.expand_dims(final_output, -1)   # 这里必须扩充一个维度　代表通道数
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size):
                    # conv
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                    # 卷积
                    conv = tf.nn.conv2d(cnn_input, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                    # 激活
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # 池化
                    pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name='pool')

                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, (1 - self.dropout_rate))

        # full + output
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # 下面两部训练时不需要 这只是测试的时候为了给出最终的结果
            self.softmax_data = tf.nn.softmax(self.scores, name="cf_softmax")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,
                                                                            labels=self.input_y,
                                                                            name="cross_entropy")
            vars = tf.trainable_variables()
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_reg_lambda

            self.loss = tf.reduce_mean(self.cross_entropy + self.l2_losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
