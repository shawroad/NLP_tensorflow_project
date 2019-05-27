"""

@file   : data_generator.py

@author : xiaolu

@time1  : 2019-05-27

"""
import numpy as np
import keras

class LMDataGenerator(keras.utils.Sequence):
    # 对数据的处理
    def __len__(self):
        # 看一下能做多少批次  所有数据量除以一批次的数据量
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __init__(self, corpus, vocab, sentence_maxlen=100, token_maxlen=50, batch_size=32, shuffle=True, token_encoding='word'):
        """Compiles a Language Model RNN based on the given parameters
        :param corpus: filename of corpus 语料的路径
        :param vocab: filename of vocabulary   语料对应的词表路径
        :param sentence_maxlen: max size of sentence  # 每个句子的最大长度
        :param token_maxlen: max size of token in characters  # 每个单词的最大长度
        :param batch_size: number of steps at each batch   # 搞多少批次
        :param shuffle: True if shuffle at the end of each epoch   # 打乱
        :param token_encoding: Encoding of token, either 'word' index or 'char' indices  # 字符级别还是词级别
        :return: Nothing
        """

        self.corpus = corpus
        # # 打开词表读取单词  .[0]是词    .[1]指的是词的标号
        self.vocab = {line.split()[0]: int(line.split()[1]) for line in open(vocab, encoding='utf8').readlines()}
        self.sent_ids = corpus
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sentence_maxlen = sentence_maxlen
        self.token_maxlen = token_maxlen
        self.token_encoding = token_encoding
        # 打开语料
        with open(self.corpus, encoding='utf8') as fp:
            self.indices = np.arange(len(fp.readlines()))
            newlines = [index for index in range(0, len(self.indices), 2)]
            self.indices = np.delete(self.indices, newlines)  # 只取了部分数据

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        batch_indices = self.indices[index*self.batch_size: (index + 1) * self.batch_size]

        # 读样本序列   也就是一条是一个句子  读取len(batch_indices)条
        word_indices_batch = np.zeros((len(batch_indices), self.sentence_maxlen), dtype=np.int32)
        if self.token_encoding == 'char':

            word_char_indices_batch = np.full((len(batch_indices), self.sentence_maxlen, self.token_maxlen), 260, dtype=np.int32)

        for i, batch_id in enumerate(batch_indices):
            # Read sentence (sample)
            word_indices_batch[i] = self.get_token_indices(sent_id=batch_id)


    def get_token_indices(self, sent_id: int):
        # 将一句话中所有的单词转为对应的id
        with open(self.corpus, encoding='utf8') as fp:
            for i, line in enumerate(fp):
                if i == sent_id:
                    token_ids = np.zeros((self.sentence_maxlen,), dtype=np.int32)
                    # Add begin of sentence index
                    token_ids[0] = self.vocab['<bos>']
                    # 遍历当前句子的每个单词
                    for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]):
                        # 将单词转为小写 看其是否在我们的词表中
                        if token.lower() in self.vocab:
                            # 单词如果在词表中 将其id加入到token_ids中
                            token_ids[j + 1] = self.vocab[token.lower()]
                        else:
                            # 否则  添加一个不知的字符串
                            token_ids[j + 1] = self.vocab['<unk>']

                        # Add end of sentence index
                    if token_ids[1]:   # 这个判断是代表你当前是有单词加入的  那我们就在其后加结束标志
                        token_ids[j + 2] = self.vocab['<eos>']
                    return token_ids


    def get_token_char_indices(self, sent_id: int):
        # 将每个单词中的每个字符转为id序列
        def convert_token_to_char_ids(token, token_maxlen):
            bos_char = 256  # <begin sentence>   句子的开始
            eos_char = 257  # <end sentence>    句子的结束
            bow_char = 258  # <begin word>   单词的开始
            eow_char = 259  # <end word>    单词的结束
            pad_char = 260  # <pad char>  填充的标志

            # 先初始化  每个单词的长度都是token_maxlen
            char_indices = np.full([token_maxlen], pad_char, dtype=np.int32)

            # Encode word to UTF-8 encoding  编码每个单词
            word_encoded = token.encode('utf-8', 'ignore')[:(token_maxlen - 2)]
            # Set characters encodings
            # Add begin of word char index
            char_indices[0] = bow_char    # 每个单词的开始 是开始的标志 bow_char
            if token == '<bos>':   # 如果这个单词是句子的开始  则将其字符列表写成bos_char
                char_indices[1] = bos_char
                k = 1
            elif token == '<eos>':  # 如果这个单词是句子的结束  则将其字符列表写成eos_char
                char_indices[1] = eos_char
                k = 1
            else:
                # Add word char indices
                for k, chr_id in enumerate(word_encoded, start=1):
                    char_indices[k] = chr_id + 1
            # Add end of word char index
            char_indices[k + 1] = eow_char

            # 简单讨论一下这一步的输出
            # 如果当前单词是<bos>说明是句子的开始 编码变为[bow_char对应的id, bos_char对应的id, eow_char对应的id, 其余都是填充的id]
            # 如果当前单词不是开始和结束的标志  编码变为[bow_char对应的id, 单词中每个字符对应的id....., eow_char对应的id, 如果还有剩余的空则是填充的id]
            # 如果当前单词是<eow>说明是句子的结束 编码变为[bow_char对应的id, eos_char对应的id, eow_char对应的id, 其余都是填充的id]

            return char_indices

        with open(self.corpus, encoding='utf8') as fp:   # 打开语料
            for i, line in enumerate(fp):
                if i == sent_id:
                    # 当前的一句话
                    token_ids = np.zeros((self.sentence_maxlen, self.token_maxlen), dtype=np.int32)
                    # Add begin of sentence char indices  在这句话的开始加bos
                    token_ids[0] = convert_token_to_char_ids('<bos>', self.token_maxlen)
                    # Add tokens' char indices   将里面每个单词中的字母变为对应的id
                    for j, token in enumerate(line.split()[:self.sentence_maxlen - 2]):
                        token_ids[j + 1] = convert_token_to_char_ids(token, self.token_maxlen)
                    # Add end of sentence char indices
                    if token_ids[1]:   # 给这句话加结尾的标志
                        token_ids[j + 2] = convert_token_to_char_ids('<eos>', self.token_maxlen)

        return token_ids

















