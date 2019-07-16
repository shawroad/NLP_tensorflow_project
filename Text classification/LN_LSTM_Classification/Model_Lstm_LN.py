"""

@file  : Model_Lstm_LN.py

@author: xiaolu

@time  : 2019-07-16

"""
import tensorflow as tf


def layer_norm_all(h, base, num_units, scope):
    # layer_norm_all(concat, 4, h_size, 'ln')
    # 把隐层单元扩展为原来的四倍, 然后将隐层进行求均值, 方差等. 归一化
    # 这里是给整体进行归一化
    with tf.variable_scope(scope):
        h_reshape = tf.reshape(h, [-1, base, num_units])
        mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
        var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
        epsilon = tf.constant(1e-3)
        rstd = tf.rsqrt(var + epsilon)
        h_reshape = (h_reshape - mean) * rstd
        h = tf.reshape(h_reshape, [-1, base * num_units])
        alpha = tf.get_variable('layer_norm_alpha', [4 * num_units],
                                initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        bias = tf.get_variable('layer_norm_bias', [4 * num_units],
                               initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        return (h * alpha) + bias


def layer_norm(x, scope='layer_norm', alpha_start=1.0, bias_start=0.0):
    # layer_norm(new_c, 'ln_c')
    # 对当个向量进行归一化
    with tf.variable_scope(scope):
        num_units = x.get_shape().as_list()[1]
        alpha = tf.get_variable('alpha', [num_units], initializer=tf.constant_initializer(alpha_start), dtype=tf.float32)
        bias = tf.get_variable('bias', [num_units], initializer=tf.constant_initializer(bias_start), dtype=tf.float32)
        mean, variance = moments_for_layer_norm(x)
        y = (alpha * (x - mean)) / (variance) + bias


def moments_for_layer_norm(x, axes=1, name=None):
    # 算输入向量的均值和方差
    epsilon = 1e-3
    if not isinstance(axes, list):
        axes = [axes]
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims=True) + epsilon)
    return mean, variance


def zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training):
    # new_h, new_c = zoneout(new_h, new_c, h, c, self.zoneout_keep_h, self.zoneout_keep_c, self.is_training)
    mask_c = tf.ones_like(c)    # 生成同等规格的全1向量
    mask_h = tf.ones_like(h)
    if is_training:
        mask_c = tf.nn.dropout(mask_c, c_keep)
        mask_h = tf.nn.dropout(mask_h, h_keep)
    mask_c *= c_keep
    mask_h *= h_keep
    h = new_h * mask_h + (-mask_h + 1.) * h
    c = new_c * mask_c + (-mask_c + 1.) * c
    return h, c


class LN_LSTMCell(tf.contrib.rnn.RNNCell):
    # 对LSTM的单元进行归一化
    def __init__(self, num_units, f_bias=1.0, use_zoneout=False,
                 zoneout_keep_h=0.9, zoneout_keep_c=0.5,
                 is_training=True, reuse=None, name=None):
        super(LN_LSTMCell, self).__init__(_reuse=reuse, name=name)
        self.num_units = num_units
        self.f_bias = f_bias
        self.use_zoneout = use_zoneout
        self.zoneout_keep_h = zoneout_keep_h    # 对这个输出执行的drop率
        self.zoneout_keep_c = zoneout_keep_c    # 对这个输出执行的drop率
        self.is_training = is_training

    def build(self, inputs_shape):
        # 定义可训练的参数  即 输入到因此的参数矩阵, 上层输出到本层隐层的参数矩阵
        w_init = tf.orthogonal_initializer(1.0)
        h_init = tf.orthogonal_initializer(1.0)
        b_init = tf.constant_initializer(0.0)
        h_size = self.num_units  # 隐层的输出有units个数决定
        self.W_xh = tf.get_variable('W_xh', [inputs_shape[1], 4 * h_size], initializer=w_init, dtype=tf.float32)
        self.W_hh = tf.get_variable('W_hh', [h_size, 4 * h_size], initializer=h_init, dtype=tf.float32)
        self.bias = tf.get_variable('bias', [4*h_size], initializer=b_init, dtype=tf.float32)


    def call(self, x, state):
        h, c = state   # LSTM中每步两个输出
        h_size = self.num_units
        concat = tf.concat(axis=1, values=[x, h])   # 将x, h 拼接
        W_full = tf.concat(axis=0, values=[self.W_xh, self.W_hh])  # 将两个参数矩阵进行对叠
        concat = tf.matmul(concat, W_full) + self.bias
        concat = layer_norm_all(concat, 4, h_size, 'ln')
        i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=concat)
        new_c = c * tf.sigmoid(f + self.f_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(layer_norm(new_c, 'ln_c')) * tf.sigmoid(o)

        if self.use_zoneout:
            # 采样dropout
            new_h, new_c = zoneout(new_h, new_c, h, c, self.zoneout_keep_h, self.zoneout_keep_c, self.is_training)
        return new_h, new_c


    def zero_state(self, batch_size, dtype):
        # 初始化的状态
        h = tf.zeros([batch_size, self.num_units], dtype=dtype)
        c = tf.zeros([batch_size, self.num_units], dtype=dtype)
        return (h, c)

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units


class Model:
    def __init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate):
        '''
        :param num_layers: 搭建多深的lstm
        :param size_layer: 每步的输出维度
        :param dimension_input: 输入的长度
        :param dimension_output: 输出的长度
        :param learning_rate: 学习率
        '''
        # 1. 定义带归一化的lstm单元
        def lstm_cell():
            return tf.contrib.rnn.LayerNormBasicLSTMCell(size_layer)

        # 2. 多层的lstm
        self.rnn_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(num_layers)])

        # 3. 占位符
        vocab_size = 4088
        self.X = tf.placeholder(tf.int32, [None, dimension_input])
        self.Y = tf.placeholder(tf.int32, [None, dimension_output])
        embedded_size = 128
        embeddings = tf.Variable(tf.random_uniform([vocab_size, embedded_size], -1, 1))
        embedded = tf.nn.embedding_lookup(embeddings, self.X)

        # 4. 带有dropout的lstm, 到此,我们的lstm定义好了
        drop = tf.contrib.rnn.DropoutWrapper(self.rnn_cells, output_keep_prob=0.5)
        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, embedded, dtype=tf.float32)

        # 5.　定义全连接层
        self.rnn_W = tf.Variable(tf.random_normal((size_layer, dimension_output)))
        self.rnn_B = tf.Variable(tf.random_normal([dimension_output]))

        self.logits = tf.matmul(self.outputs[:, -1], self.rnn_W) + self.rnn_B

        # 6. 定义损失
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        l2 = sum(0.0005 * tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.cost += l2

        # 7. define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        # 8. define acc
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


