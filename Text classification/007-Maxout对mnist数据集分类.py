"""

@file  : 007-Maxout对mnist数据集分类.py

@author: xiaolu

@time  : 2019-07-17

"""
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical


# 加载数据集　并进行预处理
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28*28)).astype('float32') / 255.
x_test = x_test.reshape((-1, 28*28)).astype('float32') / 255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def max_out(inputs, num_units, axis=None):
    # max_out可以简单理解为: 上层是100各神经元,我们将100分成50组, 每组挑一个最大的.就挑出50个,然后将其送到下一层
    # max_out(z, 50) z为上一层的输出　shape=(batch, 100), 50为这一层保存的单元
    # 可能是想从上层100维的输出中保存激活最大的50维
    # get_shape()这个函数只是用于获取输入的shape
    shape = inputs.get_shape().as_list()   # 假设shape=[128, 100]
    print(shape)
    if shape[0] is None:
        shape[0] = -1
    if axis is None:   # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]   # shape[-1] 指的是获取上层的输出
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))

    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)

    return outputs


tf.reset_default_graph()

# 1. 定义占位符
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None, 10])

# 2. 定义权重和偏执
W = tf.Variable(tf.random_normal([784, 100]))
b = tf.Variable(tf.zeros([100]))

# 3. 设置模型　
z = tf.matmul(x, W) + b    # 全连接定义 shape=(batch, 100)
# maxout = tf.reduce_max(z,axis= 1, keep_dims=True)　　# 这一步其实就实现了maxout 也就是找上一层激活的最大值
maxout = max_out(z, 50)   # 输出为[batch, 50]
W2 = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
# 构建模型
pred = tf.matmul(maxout, W2) + b2

# 计算损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

# 计算准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 参数设置
learning_rate = 0.04
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


training_epochs = 200
batch_size = 100
display_step = 1000  # 每训练1000步　显示一次

num_step = x_train.shape[0] // batch_size

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 启动循环开始训练
    for epoch in range(training_epochs):
        for step in range(num_step-1):
            batch_x, batch_y = x_train[step*batch_size: (step+1)*batch_size], y_train[step*batch_size: (step+1)*batch_size]
            _, acc, loss = sess.run([optimizer, accuracy, cost], feed_dict={x: batch_x, y: batch_y})

            if step % display_step == 0:
                print("cost:{}, acc:{}".format(loss, acc))
    print('Finish')
