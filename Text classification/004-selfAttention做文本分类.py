"""

@file   : 004-selfAttention做文本分类.py

@author : xiaolu

@time1  : 2019-05-29

"""
from keras.preprocessing import sequence
from keras.datasets import imdb
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
batch_size = 32
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dropout, Dense


# 自己实现一个层
class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        super(Self_Attention, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # 也就是初始化的那三个矩阵
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)

        # 初始化三个矩阵  为了让输入的线性变化 得到WQ, WK, WV矩阵
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)    # 一定要在最后调用它

    def call(self, x):

        # 如果将输入的所有向量合并为矩阵形式，则所有query, key, value向量也可以合并为矩阵形式表示
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ的形状:", WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        # 这里就相当于 Q*K  只不过用的是矩阵   permute_dimensions相当于将WK矩阵做了个转置  只是为了矩阵能乘在一块
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))    # 相当于把矩阵的第三维和第二维进行交换

        QK = QK / (64**0.5)    # 这里除以了64 是我自己随意的定义。 论文中说的是除以维度的数值

        QK = K.softmax(QK)  # softmax获得类似概率的值

        print("QK的形状:", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)



max_feature =  20000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)

# get_dummies就是用pd转one_hot编码  也可用to_caterierial()  这个单词可能写错了
y_train, y_test = pd.get_dummies(y_train), pd.get_dummies(y_test)

maxlen = 64

# 然后pad成同样的长度
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


S_inputs = Input(shape=(64,), dtype='int32')
embeddings = Embedding(max_feature, 128)(S_inputs)

O_seq = Self_Attention(128)(embeddings)
O_seq = GlobalAveragePooling1D()(O_seq)

O_seq = Dropout(0.5)(O_seq)
outputs = Dense(2, activation='softmax')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)

print(model.summary())

opt = Adam(lr=0.0002,decay=0.00001)
loss = 'categorical_crossentropy'
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])


h = model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test))


plt.plot(h.history["loss"], label="train_loss")
plt.plot(h.history["val_loss"], label="val_loss")
plt.plot(h.history["acc"], label="train_acc")
plt.plot(h.history["val_acc"], label="val_acc")
plt.legend()
plt.show()

model.save("imdb.h5")















