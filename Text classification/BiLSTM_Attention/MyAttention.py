"""

@file  : MyAttention.py

@author: xiaolu

@time  : 2019-07-19

"""
import tensorflow as tf
import numpy as np


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    # 注意力机制来自本篇文章　http://www.aclweb.org/anthology/N16-1174
    '''
    :param inputs: bilstm的输出　　shape=(batch_size, max_time, 2*h_size)
    :param attention_size: 超参数 Linear size of the Attention weights
    :param time_major: 若是False:则输入的形状为shape=[batch_size, max_time, cell.output_size]
    :param time_major: 若是True:则输入的形状为shape=[max_time, batch_size, cell.output_size]
    :param return_alphas: 是否输出每步的attention coefficients
    :return: RNN: [batch_size, cell.output_size] BiRNN: [batch_size, cell_fw.output_size + cell_bw.output_size]
    '''
    if isinstance(inputs, tuple):
        # 这里是为了判断是RNN还是BiRNN的输出
        inputs = tf.concat(inputs, 2)  # 是BiRNN的输出　我们需要将前向和后向输出拼接

    if time_major:
        # (max_time, Batch_size, h_size) => (Batch_size, max_time, h_size)
        inputs = tf.transpose(inputs, [1, 0, 2])

    # sequence_length = tf.shape(inputs)[1].value  # 获取序列长度 inputs.shape[1].value
    # hidden_size = tf.shape(inputs)[2].value   # inputs.shape[2].value   # 获取每步输出的维度　即:2*hidden_size
    # sequence_length = inputs.shape[1].value
    # hidden_size = inputs.shape[2].value
    # print(inputs)
    #
    # sequence_length = inputs.get_shape().as_list()[1]
    # hidden_size = inputs.get_shape().as_list()[2]
    # print(inputs.get_shape().as_list())
    #
    sequence_length = len(inputs)
    hidden_size = 256



    # 注意力机制
    W = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    # 每个时间步的输出经过全连接
    # in: [batch_size, max_time, 2*hidden_size]  out: [batch_size, max_time, attention_size]
    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W) + tf.reshape(b, [1, -1]))

    # in: [batch_size, max_time, attention_size] out: [batch_size, max_time,  1]
    vu = tf.matmul(v, tf.reshape(u, [-1, 1]))

    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])    # equal softmax

    # 输出
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output

    else:
        return output, alphas
