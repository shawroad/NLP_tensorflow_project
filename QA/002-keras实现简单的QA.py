"""

@file   : 002-keras实现简单的QA.py

@author : xiaolu

@time1  : 2019-05-30

"""
from functools import reduce
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dropout
from keras.layers import LSTM, RepeatVector, Add, Dense
from keras.models import Model
from keras.layers import concatenate

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
# 1. 对话集 构建网络—— embedding + dropout
embedding_out = 50   # 映射成50维向量
dialog = Input(shape=(dialog_maxlen, ), dtype="int32")
encoded_dialog = Embedding(lexicon_size, embedding_out)(dialog)
encoded_dialog = Dropout(0.3)(encoded_dialog)


# 2. 问题集 embedding + dropout + lstm
lstm_out = 100    # 这里控制LSTM的输出 是因为我们等会将LSTM输出的问题向量和 对话的词嵌入向量进行相加 所以维度必须一致
question = Input(shape=(ques_maxlen, ), dtype="int32")
encoded_ques = Embedding(lexicon_size, embedding_out)(question)
encoded_ques = Dropout(0.3)(encoded_ques)
encoded_ques = LSTM(units=lstm_out)(encoded_ques)
# LSTM的输出进行RepeatVector，也就是重复dialog_maxlen次，这样encodeed_ques的shape就变为了（dialog_maxlen，lstm_out）
encoded_ques = RepeatVector(dialog_maxlen)(encoded_ques)


# 3. merge 对话集和问题集的模型 merge后进行 lstm + dropout + dense
merged = concatenate([encoded_dialog, encoded_ques])
# merged = Add()([encoded_dialog, encoded_ques])
# 上一步也可以进行拼接   merged = layers.concatenate([encoded_sentence, encoded_question])  这样输入就不需要和对话的词嵌入的唯独相同
merged = LSTM(units=lstm_out)(merged)
merged = Dropout(0.3)(merged)
preds = Dense(units=lexicon_size, activation='softmax')(merged)


model = Model([dialog, question], preds)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 开始训练
batchs = 32
epochs = 1
model.fit([dialog_train, ques_train], ans_train,
          batch_size=batchs,
          epochs=epochs,
          validation_data=([dialog_test, ques_test], ans_test))


print('Evaluation')
loss, acc = model.evaluate([dialog_test, ques_test], ans_test, batch_size=batchs)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


# 扩展 LSTM中参数的用法
# units： 输出维度
# input_dim： 输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)
# return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
# input_length： 当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接Flatten层，然后又要连接Dense层时，需要指定该参数，否则全连接的输出无法计算出来


