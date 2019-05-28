"""

@file   : 004-Glove.py

@author : xiaolu

@time1  : 2019-05-28

"""
from nltk.tokenize import word_tokenize
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim


# 参数设置
context_size = 3  # 设置窗口的大小
embed_size = 2  # 词嵌入的维度
xmax = 2
alpha = 0.75   # 以上两个参数是定义权重函数是所需要的 可以自己随意设定
batch_size = 20
l_rate = 0.001
num_epochs = 10

# 打开文件 读取语料
fr = open('short_story.txt', 'r')
text = fr.read().lower()
fr.close()

# print(text)


# 建立词表
word_list = word_tokenize(text)   # 分词
vocab = np.unique(word_list)    # 去重后的词表
w_list_size = len(word_list)   # 语料中词的个数
vocab_size = len(vocab)   # 词表的大小

# 词到id的映射
w_to_i = {word: ind for ind, word in enumerate(vocab)}
# print(w_to_i)

comat = np.zeros((vocab_size, vocab_size))
for i in range(w_list_size):
    for j in range(1, context_size+1):
        ind = w_to_i[word_list[i]]  # 将语料中每次词拿出来  转为id
        if i - j > 0:    # 找去窗口内的左边词汇id
            lind = w_to_i[word_list[i-j]]
            comat[ind, lind] += 1.0/j   # 考虑的权重  你若越远 这个权重越低  你若越近 权重越高
        if i + j < w_list_size:    # 找去窗口内的左边词汇id
            rlid = w_to_i[word_list[i+j]]
            comat[ind, rlid] += 1.0/j

print(comat)

# np.nonzero()  输出为一个元组  第一个元组是非零元素所在的行  第二个元素是非零元素所在的列
coocs = np.transpose(np.nonzero(comat))    # 现在 coocs的每一行就是非零元素所在的坐标


# 权重函数
def wf(x):
    if x < xmax:
        return (x/xmax) ** alpha
    return 1


# 设定词向量 和 偏置项
l_embed, r_embed = [
    [Variable(torch.from_numpy(np.random.normal(0, 0.01, (embed_size, 1))),
              requires_grad=True) for j in range(vocab_size)] for i in range(2)]

l_biases, r_biases = [
    [Variable(torch.from_numpy(np.random.normal(0, 0.01, 1)),
              requires_grad=True) for j in range(vocab_size)] for i in range(2)]

# 设定优化器
optimizer = optim.Adam(l_embed + r_embed + l_biases + r_biases, lr=l_rate)


# 产生批数据
def gen_batch():
    sample = np.random.choice(np.arange(len(coocs)), size=batch_size, replace=False)   # 从中选取batch_size条数据
    l_vecs, r_vecs, covals, l_v_bias, r_v_bias = [], [], [], [], []
    for chosen in sample:
        ind = tuple(coocs[chosen])   # 取出当前所选样本的坐标
        l_vecs.append(l_embed[ind[0]])
        r_vecs.append(r_embed[ind[1]])
        covals.append(comat[ind])
        l_v_bias.append(l_biases[ind[0]])
        r_v_bias.append(r_biases[ind[1]])
    return l_vecs, r_vecs, covals, l_v_bias, r_v_bias


# 模型的训练
for epoch in range(num_epochs):
    num_batches = int(w_list_size/batch_size)   # 看一下一批需去多少数据
    avg_loss = 0.0
    for batch in range(num_batches):
        optimizer.zero_grad()
        l_vecs, r_vecs, covals, l_v_bias, r_v_bias = gen_batch()
        # 定义损失函数
        # For pytorch v2 use, .view(-1) in torch.dot here. Otherwise, no need to use .view(-1).
        loss = sum([torch.mul((torch.dot(l_vecs[i].view(-1), r_vecs[i].view(-1))
                               + l_v_bias[i] + r_v_bias[i] - np.log(covals[i]))**2, wf(covals[i])) for i in range(batch_size)])

        avg_loss += loss.data[0]/num_batches
        loss.backward()   # 反向传播
        optimizer.step()
    print("per epoch average loss:"+str(epoch+1)+": ", avg_loss)

# 这里设置的嵌入维度是2  可以进行可视化
if embed_size == 2:
    # 从词表中随机选取10个词
    word_inds = np.random.choice(np.arange(len(vocab)), size=10, replace=False)
    for word_ind in word_inds:
        # Create embedding by summing left and right embeddings
        w_embed = (l_embed[word_ind].data + r_embed[word_ind].data).numpy()
        x, y = w_embed[0][0], w_embed[1][0]
        plt.scatter(x, y)
        plt.annotate(vocab[word_ind], xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig("glove.png")
