"""

@file  : model.py

@author: xiaolu

@time  : 2019-12-19

"""
import os
import sys
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


class TextCNN:
    '''
    A CNN for text classification.
    '''
    def __init__(self, sequence_length, num_classes, vocab_size, batch_size,
                 n_gram, pixel_weight, pixel_height, num_filters, l2_reg_lambda):
        '''
        :param sequence_length:
        :param num_classes:
        :param vocab_size:
        :param batch_size:
        :param n_gram:
        :param pixel_weight:
        :param pixel_height:
        :param num_filters:
        :param l2_reg_lambda:
        '''
        self.batch_size = batch_size

        # 1. 定义输入
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_data')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_label')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # 2. 词嵌入
        with tf.name_scope('embedding'):
            self.vocab_table = tf.Variable(tf.random_uniform([vocab_size, pixel_height, pixel_weight],
                                                             -0.1, 0.1), name='W')

            self.embedded_chars = tf.nn.embedding_lookup(self.vocab_table, self.input_x)

            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # print(self.embedded_chars_expanded)  # [batch_size, sequence_length, pixel_height, pixel_weight, 1]

        # 3. 三维卷积层
        with tf.name_scope("conv_layer"):
            filter_shape = [n_gram, pixel_height, pixel_weight, 1, num_filters]
            filters = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
            # print(filters)    # [n_gram, pixel_height, pixel_weight,1, num_filters]

            conv3d_net = tf.nn.conv3d(self.embedded_chars_expanded, filters, strides=[1, 1, 1, 1, 1],
                                      padding='VALID', name='conv3d')
            # print(conv3d_net)  # [batch_size, sequence_length, pixel_height, pixel_weight, num_filters]

            conv3d_net_reshape = tf.reshape(conv3d_net,
                                            [-1, conv3d_net.shape[1]*conv3d_net.shape[2]*conv3d_net.shape[3], conv3d_net.shape[4]])
            # print(conv3d_net_reshape)  # [batch_size, ***, num_filters]

            # 给卷积加上偏置
            bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bias")
            conv3d_relu = tf.nn.relu(tf.nn.bias_add(conv3d_net_reshape, bias), name="relu")

            print(conv3d_relu)  # [batch_size, ** *, num_filters]

            conv3d_expdim = tf.expand_dims(conv3d_relu, -1)

            # 接下来 跟一个空洞卷积
            dilation_filter_shape = [3, 1, conv3d_expdim.shape[3].value]
            dilation_filter = tf.Variable(tf.truncated_normal(dilation_filter_shape,
                                                              stddev=0.1),
                                                              name="dilation_filter")
            print(dilation_filter)

            dilated_conv3d = tf.nn.dilation2d(
                input=conv3d_expdim,
                filter=dilation_filter,
                strides=[1, 1, 1, 1],
                rates=[1, 3, 1, 1],
                padding="VALID",
                name="dilation")
            print(dilated_conv3d)

            pooled_net = tf.nn.max_pool2d(
                dilated_conv3d,
                ksize=[1, 3, 1, 1],
                strides=[1, 3, 1, 1],
                padding="VALID",
                name="pool"
            )
            print(pooled_net)

            pooled_drop = tf.nn.dropout(pooled_net, self.dropout_keep_prob)
            pooled_drop_flat = tf.reshape(pooled_drop, [-1, pooled_drop.shape[1] * pooled_drop.shape[2] * pooled_drop.shape[3]])
            print(pooled_drop_flat)

        # Creat 3*FC layer
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.05)):
            fc_net1 = slim.fully_connected(pooled_drop_flat, 1250, scope='fc1')
            fc_net1 = slim.dropout(fc_net1, self.dropout_keep_prob, scope='fc_drop1')
            fc_net2 = slim.fully_connected(fc_net1, 512, scope='fc2')
            fc_net2 = slim.dropout(fc_net2, self.dropout_keep_prob, scope='fc_drop2')
            fc_net3 = slim.fully_connected(fc_net2, 100, scope='fc3')

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[100, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.scores = tf.nn.xw_plus_b(fc_net3, W, b, name="scores")
            self.softmax_data = tf.nn.softmax(self.scores, name="cf_softmax")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss and l2_loss
        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores,
                                                                         labels=self.input_y,name="cross_entropy")
            # 权重正则化
            vars = tf.trainable_variables()
            self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_reg_lambda

            self.loss = tf.reduce_mean(self.cross_entropy + self.l2_losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
