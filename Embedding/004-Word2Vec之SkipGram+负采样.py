"""

@file   : 004-Word2Vec之SkipGram+负采样.py

@author : xiaolu

@time   : 2019-07-07

"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.reset_default_graph()


sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]


# 构造词表
word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}


batch_size = 20
embedding_size = 2   # 嵌入到二维  为了可视化
num_sampled = 10   # for negative sampling, less than batch_size
voc_size = len(word_list)


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels

# skip gram 构造 目标词 到 标签的映射关系  这里的窗口设置为1  只取上下文各一个词
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_grams.append([target, w])


# 定义输入
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])   # 标签整成[batch_size, 1] 为了用tf.nn.nce_loss,

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))


# 定义损失  以及优化器
cost = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


# 训练
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(5000):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)   # 获取批量数据
        _, loss = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs, labels: batch_labels})

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    trained_embeddings = embeddings.eval()


for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()