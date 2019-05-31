"""

@file   : 003-keras实现QA版本2.py

@author : xiaolu

@time1  : 2019-05-31

"""
from functools import reduce
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dropout
from keras.layers import LSTM, RepeatVector, Add, Dense
from keras.models import Sequential, Model
from keras.layers import concatenate
from keras.layers import dot
from keras.layers import Activation, add, Permute

# parse_dialog 将所有的对话进行解析，返回tokenize后的(对话,问题,答案)
# 如果 only_supporting为真表明只返回含有答案的对话
def parse_dialog(lines, only_supporting=False):
    data = []
    dialog = []
    for line in lines:
        nid, line = line.split(" ", 1)
        nid = int(nid)
        # 标号为1表示新的一段文本的开始，重新记录
        if nid == 1:
            dialog = []
        # 含有tab键的说明就是问题，将问题，答案和答案的索引分割开
        if '?' in line:
            ques, ans, data_idx = line.split('\t')
            # 将问题分词
            ques = ques.replace('?', "").split(" ")
            substory = None
            if only_supporting:
                substory = dialog[data_idx]   # 相当于只把含有答案的那句话拿出来
            else:
                substory = [x for x in dialog]  # 是把所有的对话拿出来
            data.append((substory, ques, ans))
        else:
            # 不含有tab键的就是对话，tokenize后加入dialog的list
            line = line.replace('\n', '').split(' ')
            dialog.append(line)  # 将对话进行解析

    return data


# 这里的maxlen是控制文本最大长度的，可以利用分位数找出覆盖90%数据的长度，令其为maxlen。
# 否则序列长度太长，训练时内存不够。
def get_dialog(f, only_supporting=False, max_length=None):
    # 将对话完整的取出来
    data = parse_dialog(f.readlines(), only_supporting=only_supporting)
    # flatten将两个列表拉直在一块, 我们打算每篇对话的内容拉成一个列表
    flatten = lambda data: reduce(lambda x, y: x + y, data)

    data = [(flatten(dialog), ques, ans) for (dialog, ques, ans) in data
            if not max_length or len(flatten(dialog)) < max_length]
    return data


# 将数据转为id序列 然后pad成一样的长度
def vectorize_dialog(data, wd_idx, dialog_maxlen, ques_maxlen):
    dialog_vec = []
    ques_vec = []
    ans_vec = []
    for dialog, ques, ans in data:
        dialog_idx = [wd_idx[w] for w in dialog]
        ques_idx = [wd_idx[w] for w in ques]

        ans_zero = np.zeros(len(wd_idx) + 1)
        ans_zero[wd_idx[ans]] = 1   # 建一个词表一样的长度向量，将答案那一维置为1

        dialog_vec.append(dialog_idx)
        ques_vec.append(ques_idx)
        ans_vec.append(ans_zero)
    # 进行序列的pad
    dialog_vec = pad_sequences(dialog_vec, maxlen=dialog_maxlen)
    ques_vec = pad_sequences(ques_vec, maxlen=ques_maxlen)
    ans_vec = np.array(ans_vec)

    return dialog_vec, ques_vec, ans_vec


# 读取 训练数据  测试数据
ftrain = open('./data/qa5_three-arg-relations_train.txt', 'r', encoding='utf8')
ftest = open('./data/qa5_three-arg-relations_train.txt', 'r', encoding='utf8')

train = get_dialog(ftrain)
test = get_dialog(ftest)

# e = len(train + test)
# print(e)  # 20000  有两万组对话


# 建立词表
lexicon = set()
for dialog, ques, ans in train + test:
    # 将三种词序列加起来 去重 词表就构造好了
    lexicon |= set(dialog + ques + [ans])

lexicon = sorted(lexicon)
lexicon_size = len(lexicon) + 1
# print(lexicon_size)   # 48  没看错 两万多组对话 只有48个单

wd_idx = dict((wd, idx+1) for idx, wd in enumerate(lexicon))
dialog_maxlen = max(map(len, (x for x, _, _ in train+test)))  # 获取对话的最长度
ques_maxlen = max(map(len, (x for _, x, _ in train+test)))  # 获取问题的最长度


# #计算分位数，在get_dialog函数中传参给max_len
# dia_80 = np.percentile(map(len, (x for x, _, _ in train + test)), 80)  # 保留80%的数据长度多少合适
# dia_80 = int(dia_80)


# 训练集和测试集进行id映射   也就是将文本转为数字
dialog_train, ques_train, ans_train = vectorize_dialog(train, wd_idx, dialog_maxlen, ques_maxlen)
dialog_test, ques_test, ans_test = vectorize_dialog(test, wd_idx, dialog_maxlen, ques_maxlen)


# 建立模型
input_sequence = Input((dialog_maxlen,))
question = Input((ques_maxlen,))

# 对话词嵌入
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=lexicon_size, output_dim=64))
input_encoder_m.add(Dropout(0.3))

# 对话词嵌入   这两次对话词嵌入
input_encoder_c = Sequential()
# 注意这里的输出维度  我们打算让其输出和问题的词嵌入进行相乘  所以这里的维度需要和问题的长度一致
input_encoder_c.add(Embedding(input_dim=lexicon_size, output_dim=ques_maxlen))
input_encoder_c.add(Dropout(0.3))

# 问题词嵌入
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=lexicon_size, output_dim=64))
question_encoder.add(Dropout(0.3))

input_encoder_m = input_encoder_m(input_sequence)
input_encoder_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

#  试着想一下  如果两个向量相似 对应相乘会比较大， 若不相似 对应长度可能不够大  这里的思想应该是放大含问题的对话 减小与问题无关的对话权重
match = dot([input_encoder_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# 在对话中把问题融进去
response = add([match, input_encoder_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)   # (samples, query_maxlen, story_maxlen)

# 拼接
answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)  # (samples, 32)

answer = Dropout(0.3)(answer)

answer = Dense(units=lexicon_size, activation='softmax')(answer)  # (samples, vocab_size)


# 编译模型
model = Model([input_sequence, question], answer)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batchs = 32
epochs = 1
model.fit([dialog_train, ques_train], ans_train,
          batch_size=batchs,
          epochs=epochs,
          validation_data=([dialog_test, ques_test], ans_test))

# 保存模型
model.save('memm_qa.h5')



