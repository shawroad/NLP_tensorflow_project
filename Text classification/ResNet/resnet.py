"""

@file  : resnet.py

@author: xiaolu

@time  : 2019-07-29

"""
import tensorflow as tf


class Model(object):
    def __init__(self, sequence_length, num_classes, vocab_size, batch_size, embedding_size,
                 mode, init_size, residual_num, l2_reg_lambda=0.0, dropout_keep_prob=1.0):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.mode = mode
        self.init_size = init_size
        self.residual_num = residual_num
        self.l2_reg_lambda = l2_reg_lambda
        self.dropout_keep_prob = dropout_keep_prob

        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_data')
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='input_label')

        self._build_model()

    def _build_model(self):

        # 1. embedding
        with tf.name_scope('embedding'):
            x = self.input_x
            x = self._embedd(x)
            # if self.mode == 'train':
            #     x = self._dropout(x)
            print('embedding:x.shape', x.shape.as_list())

        # resnet_block
        with tf.name_scope('init_fc'):
            x = self._fully_connected('init_fc', x, self.init_size)
            print("init_fc:x.shape:", x.shape.as_list())
            if self.mode == 'train':
                x = self._dropout(x)

        for i in range(self.residual_num):
            name = 'residual_' + str(i)
            with tf.name_scope(name):
                x = self._residual_fc(name, x)
        print("residual:x.shape:", x.shape.as_list())

        # output
        with tf.name_scope('logit'):
            self.score = self._fully_connected('fc', x, self.num_classes)
            # if self.mode == 'train':
            #     self.score = self._dropout(self.score)
            self.logit = tf.nn.softmax(self.score, name='logit')
            self.prediction = tf.argmax(self.score, 1, name='prediction')

        # define loss and optimizer
        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.score,
                                                                         labels=self.input_y, name="cross_entropy")
            vars = tf.trainable_variables()
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v)
                                       for v in vars]) * self.l2_reg_lambda

            self.loss = tf.reduce_mean(self.cross_entropy + self.l2_losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

        # define accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")

    def _embedd(self, x):
        self.embedd_W = tf.Variable(tf.random_uniform(
            [self.vocab_size, self.embedding_size], -0.1, 0.1, name='embedd_W'
        ))
        x = tf.nn.embedding_lookup(self.embedd_W, x)
        x = tf.expand_dims(x, -1)
        return x

    def _residual_fc(self, name, x):
        """
        Function: _residual_fc
        Summary: A implementation of the mothod in the paper
        <ResNet with one-neuron hidden layers is a Universal Approximator>
        Examples: y = VReLU(ux + b)
        """
        org_x = x
        with tf.variable_scope(name):
            u = tf.get_variable('U', [x.shape[1], 1], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.get_variable('b', [1], initializer=tf.constant_initializer())
            v = tf.get_variable('V', [1, x.shape[1]], initializer=tf.truncated_normal_initializer(stddev=0.1))

            x = tf.nn.xw_plus_b(x, u, b)
            x = tf.nn.relu(x)
            x = tf.matmul(x, v)
            if self.mode == 'train':
                x = self._dropout(x)
        x = x + org_x

        return x

    def _fully_connected(self, name, x, out_dim):
        ndims = len(x.shape.as_list())
        print("_fully_connected, x.ndims::", ndims)
        with tf.variable_scope(name):
            if ndims == 4:
                x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]*x.shape[3]])
            w = tf.get_variable('FW', [x.shape[1], out_dim],
                                initializer=tf.truncated_normal_initializer(stddev=0.05))
            b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())

            return tf.nn.xw_plus_b(x, w, b)


    def _dropout(self, x):

        return tf.nn.dropout(x, self.dropout_keep_prob)

