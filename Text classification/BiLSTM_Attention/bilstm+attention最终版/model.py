"""

@file  : model.py

@author: xiaolu

@time  : 2019-07-30

"""
import tensorflow as tf


class Model:
    def __init__(self, embedding_size, rnn_size, layer_size, vocab_size, attn_size, sequence_length,
                 n_classes, batch_size, l2_reg_lambda=0.0):

        self.batch_size = batch_size
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_data')
        self.input_y = tf.placeholder(tf.float32, [None, n_classes], name='input_label')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # inputs shape : (batch_size , sequence_length , rnn_size)
            # bidirection rnn 的inputs shape 要求是(sequence_length, batch_size, rnn_size)
            # 因此这里需要对inputs做一些变换
            # 经过transpose的转换已经将shape变为(sequence_length, batch_size, rnn_size)
            # 只是双向rnn接受的输入必须是一个list,因此还需要后续两个步骤的变换
            inputs = tf.transpose(self.embedded_chars, [1, 0, 2])
            # 转换成(batch_size * sequence_length, rnn_size)
            inputs = tf.reshape(inputs, [-1, rnn_size])
            # 转换成list,里面的每个元素是(batch_size, rnn_size)
            inputs = tf.split(inputs, sequence_length, 0)

        # 定义前向RNN Cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            print(tf.get_variable_scope().name)
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list),
                                                           output_keep_prob=self.dropout_keep_prob)


        # 定义反向RNN Cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            print(tf.get_variable_scope().name)
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list),
                                                           output_keep_prob=self.dropout_keep_prob)

        # 定义双向lstm
        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs,
                                                                    dtype=tf.float32)

        # 定义attention layer
        attention_size = attn_size
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2 * rnn_size, attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            for t in range(sequence_length):
                u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
                u_list.append(u_t)

            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in range(sequence_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)

            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            self.alpha = tf.nn.softmax(attn_zconcat)

            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(self.alpha, [1, 0]), [sequence_length, -1, 1])
            self.final_output = tf.reduce_sum(outputs * alpha_trans, 0)

        print(self.final_output.shape)

        # scores and predictions
        with tf.name_scope("output"):
            self.final_out = tf.reduce_sum(outputs, 0)
            W = tf.get_variable(
                "W",
                shape=[2*rnn_size, n_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(self.final_out, W, b, name="scores")

            self.softmax_data = tf.nn.softmax(self.scores, name="logit")

            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss and l2_loss
        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                         labels=self.input_y,
                                                                         name="cross_entropy")

            vars = tf.trainable_variables()
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_reg_lambda

            self.loss = tf.reduce_mean(self.cross_entropy + self.l2_losses)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")