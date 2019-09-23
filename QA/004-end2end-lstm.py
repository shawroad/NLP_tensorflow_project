"""

@file  : 004-end2end-lstm.py

@author: xiaolu

@time  : 2019-09-19

"""
import tensorflow as tf
import numpy as np
from copy import deepcopy


class BaseDataLoader:
    # 这个类只是定义了一些参数 为了后面想要某个属性的时候方便写
    def __init__(self):
        self.data = {
            'size': None,
            'val': {
                'inputs': None,
                'questions': None,
                'answers': None,
            },
            'len': {
                'inputs_len': None,
                'inputs_sent_len': None,
                'questions_len': None,
                'answers_len': None
            }
        }
        self.vocab = {
            'size': None,
            'word2idx': None,
            'idx2word': None,
        }
        self.params = {
            'vacab_size': None,
            '<start>': None,
            '<end>': None,
            'max_input_len': None,
            'max_sent_len': None,
            'max_quest_len': None,
            'max_answer_len': None,
        }


class DataLoader(BaseDataLoader):
    def __init__(self, path, is_training, vocab=None, params=None):
        super(DataLoader, self).__init__()
        data, lens = self.load_data(path)
        if is_training:
            # 若是训练  得先建立词表
            self.build_vocab(data)
        else:
            self.demo = data
            self.vocab = vocab
            self.params = deepcopy(params)
        self.is_training = is_training
        # 进行pad操作
        self.padding(data, lens)

    def load_data(self, path):
        # 对三种数据进行简单的分开  对话  问题  答案
        data, lens = bAbI_data_load(path)
        self.data['size'] = len(data[0])
        return data, lens

    def build_vocab(self, data):
        signals = ['<pad>', '<unk>', '<start>', '<end>']  # 四种标记
        inputs, questions, answers = data   # 对话, 问题, 答案  三者的词表
        i_words = [w for facts in inputs for fact in facts for w in fact if w != '<end>']
        q_words = [w for question in questions for w in question]
        a_words = [w for answer in answers for w in answer if w != '<end>']
        words = list(set(i_words + q_words + a_words))   # 总的去重后的词表
        self.params['vocab_size'] = len(words) + 4   # 总的词表长度   +4是因为有四个特殊标记
        # 给特殊标记赋索引
        self.params['<start>'] = 2
        self.params['<end>'] = 3
        # 词表和id的映射
        self.vocab['word2idx'] = {word: idx for idx, word in enumerate(signals + words)}
        self.vocab['idx2word'] = {idx: word for word, idx in self.vocab['word2idx'].items()}

    def padding(self, data, lens):
        # 获取那四种长度
        inputs_len, inputs_sent_len, questions_len, answers_len = lens

        self.params['max_input_len'] = max(inputs_len)
        self.params['max_sent_len'] = max([fact_len for batch in inputs_sent_len for fact_len in batch])
        self.params['max_quest_len'] = max(questions_len)
        self.params['max_answer_len'] = max(answers_len)

        self.data['len']['inputs_len'] = np.array(inputs_len)   # 有多长个场景对话
        for batch in inputs_sent_len:
            # 对话句子的最长度 减去每句话的最长度 就是要填充的长度  补0就是补pad
            batch += [0] * (self.params['max_input_len'] - len(batch))
        self.data['len']['inputs_sent_len'] = np.array(inputs_sent_len)
        self.data['len']['questions_len'] = np.array(questions_len)
        self.data['len']['answers_len'] = np.array(answers_len)

        inputs, questions, answers = deepcopy(data)
        for facts in inputs:
            for sentence in facts:
                for i in range(len(sentence)):
                    # 将每句话转为id序列  长度不够的进行填充
                    sentence[i] = self.vocab['word2idx'].get(sentence[i], self.vocab['word2idx']['<unk>'])
                sentence += [0] * (self.params['max_sent_len'] - len(sentence))

            # 每个场景的对话条数不一致  我们要把每个场景的对话条数也进行padding
            paddings = [0] * self.params['max_sent_len']
            facts += [paddings] * (self.params['max_input_len'] - len(facts))

        for question in questions:
            for i in range(len(question)):
                question[i] = self.vocab['word2idx'].get(question[i], self.vocab['word2idx']['<unk>'])
            question += [0] * (self.params['max_quest_len'] - len(question))

        for answer in answers:
            for i in range(len(answer)):
                answer[i] = self.vocab['word2idx'].get(answer[i], self.vocab['word2idx']['<unk>'])

        self.data['val']['inputs'] = np.array(inputs)
        self.data['val']['questions'] = np.array(questions)
        self.data['val']['answers'] = np.array(answers)


def bAbI_data_load(path, END=['<end>']):
    inputs = []
    questions = []
    answers = []

    inputs_len = []
    inputs_sent_len = []
    questions_len = []
    answers_len = []

    for d in open(path):
        index = d.split(' ')[0]   # 每行的索引  每次从1开始代表的是一个新语料
        if index == '1':
            fact = []
        if '?' in d:    # 代表这行是一个问题
            temp = d.split('\t')   # 把问题和答案分离 因为我们的答案和问题在一行
            q = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']  # 问题取出来 然后按空格分类 由每个词组成的列表
            a = temp[1].split() + END
            fact_copied = deepcopy(fact)    # 把之前没有问号的当做阅读吧
            # fact_copied 格式   [[对话的第一行词列表], [对话的第二行词列表], [*]....]
            inputs.append(fact_copied)
            questions.append(q)
            answers.append(a)

            inputs_len.append(len(fact_copied))    # 一组对话总共有几句
            inputs_sent_len.append([len(s) for s in fact_copied])  # 每句对话的长度
            questions_len.append(len(q))
            answers_len.append(len(a))
        else:
            tokens = d.replace('.', '').replace('\n', '').split(' ')[1:] + END   # 将每句没有问号的当做输入， 这里将每句话按词组成列表
            fact.append(tokens)

    # 返回说明: [对话词表, 问题词表, 答案词表], [总共有几组对话(几个场景对话), 每组对话中每句对话的长度, 问题长度, 答案长度]
    return [inputs, questions, answers], [inputs_len, inputs_sent_len, questions_len, answers_len]


def hop_forward(question, memory_o, memory_i, response_proj, inputs_len, questions_len, is_training):
    '''
    :param question: 问题
    :param memory_o: 对话的词嵌入部分1
    :param memory_i: 对话的词嵌入部分2
    :param response_proj: 传进来是一个dense 是将回答和问题concat然后dense
    :param inputs_len: 对话的长度
    :param questions_len: 问题长度
    :param is_training:
    :return:
    '''
    match = tf.matmul(question, memory_i, transpose_b=True)  # 问题与对话进行揉搓
    match = pre_softmax_masking(match, inputs_len)  # 加入mask

    match = tf.nn.softmax(match)  # 当前问题与对话中每句话的匹配程度

    match = post_softmax_masking(match, questions_len)

    response = tf.matmul(match, memory_o)
    return response_proj(tf.concat([response, question], -1))


def pre_softmax_masking(x, seq_len):
    '''
    加入mask
    :param x: 问题与对话进行揉搓的结果
    :param seq_len:
    :return:
    '''
    paddings = tf.fill(tf.shape(x), float('-inf'))
    T = tf.shape(x)[1]
    max_seq_len = tf.shape(x)[2]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, 1), [1, T, 1])
    return tf.where(tf.equal(masks, 0), paddings, x)


def post_softmax_masking(x, seq_len):
    T = tf.shape(x)[2]
    max_seq_len = tf.shape(x)[1]
    masks = tf.sequence_mask(seq_len, max_seq_len, dtype=tf.float32)
    masks = tf.tile(tf.expand_dims(masks, -1), [1, 1, T])
    return (x * masks)


def shift_right(x):
    batch_size = tf.shape(x)[0]
    start = tf.to_int32(tf.fill([batch_size, 1], START))
    return tf.concat([start, x[:, :-1]], 1)


def embed_seq(x, vocab_size, zero_pad=True):
    '''
    词嵌入
    :param x: 待词嵌入的语料
    :param vocab_size: 词表大小
    :param zero_pad:
    :return:
    '''
    lookup_table = tf.get_variable('lookup_table', [vocab_size, size_layer], tf.float32)
    if zero_pad:
        # 要不要给填充的0进行词嵌入
        lookup_table = tf.concat((tf.zeros([1, size_layer]), lookup_table[1:, :]), axis=0)
    return tf.nn.embedding_lookup(lookup_table, x)


def position_encoding(sentence_size, embedding_size):
    '''
    位置编码
    :param sentence_size: 句子长度
    :param embedding_size: 词嵌入的维度
    :return: 位置编码后的tensor
    '''
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return tf.convert_to_tensor(np.transpose(encoding))


def input_mem(x, vocab_size, max_sent_len, is_training):
    '''
    对对话进行编码
    :param x: 对话
    :param vocab_size: 词表的大小
    :param max_sent_len: 对话的长度
    :param is_training:
    :return:
    '''
    x = embed_seq(x, vocab_size)
    x = tf.layers.dropout(x, dropout_rate, training=is_training)
    pos = position_encoding(max_sent_len, size_layer)
    x = tf.reduce_sum(x * pos, 2)  # 每句话词嵌入以后 将第三维度进行累加
    return x


def quest_mem(x, vocab_size, max_quest_len, is_training):
    '''
    对问题进行词嵌入
    :param x: 问题
    :param vocab_size: 词表大小
    :param max_quest_len: 问题长度
    :param is_training: 是否对0填充进行词嵌入
    :return:
    '''
    x = embed_seq(x, vocab_size)
    x = tf.layers.dropout(x, dropout_rate, training=is_training)  # 引入dropout
    pos = position_encoding(max_quest_len, size_layer)  # 加入位置向量
    return (x * pos)


class QA:
    def __init__(self, vocab_size):
        # 1. 定义输出占位符
        self.questions = tf.placeholder(tf.int32, [None, None])
        self.inputs = tf.placeholder(tf.int32, [None, None, None])
        self.answers = tf.placeholder(tf.int32, [None, None])

        self.questions_len = tf.placeholder(tf.int32, [None])
        self.inputs_len = tf.placeholder(tf.int32, [None])
        self.answers_len = tf.placeholder(tf.int32, [None])

        self.training = tf.placeholder(tf.bool)

        max_sent_len = train_data.params['max_sent_len']
        max_quest_len = train_data.params['max_quest_len']
        max_answer_len = train_data.params['max_answer_len']

        # 2. 词嵌入
        lookup_table = tf.get_variable('lookup_table', [vocab_size, size_layer], tf.float32)

        lookup_table = tf.concat((tf.zeros([1, size_layer]), lookup_table[1:, :]), axis=0)

        # 2.1 对问题进行编码
        with tf.variable_scope('questions'):
            question = quest_mem(self.questions, vocab_size, max_quest_len, self.training)

        # 2.2 对对话进行编码
        with tf.variable_scope('memory_o'):
            memory_o = input_mem(self.inputs, vocab_size, max_sent_len, self.training)

        # 2.3 对对话再次进行编码
        with tf.variable_scope('memory_i'):
            memory_i = input_mem(self.inputs, vocab_size, max_sent_len, self.training)

        # 3. 问题和对话进行揉搓
        with tf.variable_scope('interaction'):
            response_proj = tf.layers.Dense(size_layer)

            for _ in range(n_hops):
                answer = hop_forward(question, memory_o, memory_i, response_proj,
                                     self.inputs_len, self.questions_len, self.training)
                question = answer

        with tf.variable_scope('memory_o', reuse=True):
            embedding = tf.get_variable('lookup_table')

        cell = tf.nn.rnn_cell.LSTMCell(size_layer)
        vocab_proj = tf.layers.Dense(vocab_size)

        state_proj = tf.layers.Dense(size_layer)
        init_state = state_proj(tf.layers.flatten(answer))
        init_state = tf.layers.dropout(init_state, dropout_rate, training=self.training)
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=tf.nn.embedding_lookup(embedding, shift_right(self.answers)),
            sequence_length=tf.to_int32(self.answers_len)
        )
        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=init_state, h=init_state)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell,
                                                  helper=helper,
                                                  initial_state=encoder_state,
                                                  output_layer=vocab_proj)

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                 maximum_iterations=tf.shape(self.inputs)[1])

        self.outputs = decoder_output.rnn_output

        # 贪婪搜索
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                          start_tokens=tf.tile(tf.constant([START], dtype=tf.int32),
                                                                               [tf.shape(self.inputs)[0]]),
                                                          end_token=END)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=encoder_state,
            output_layer=vocab_proj
        )

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            maximum_iterations=max_answer_len
        )

        self.logits = decoder_output.sample_id
        # 计算准确率和损失
        correct_pred = tf.equal(self.logits, self.answers)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
            logits=self.outputs,
            targets=self.answers,
            weights=tf.ones_like(self.answers, tf.float32)
        ))

        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)


if __name__ == '__main__':
    # 1. 加载数据   并对数据进行预处理
    train_data = DataLoader(path="./data/qa5_three-arg-relations_train.txt", is_training=True)
    test_data = DataLoader(path='./data/qa5_three-arg-relations_test.txt', is_training=False,
                           vocab=train_data.vocab, params=train_data.params)

    START = train_data.params['<start>']
    END = train_data.params['<end>']

    # 2. 定义超参数
    epoch = 20
    batch_size = 64
    size_layer = 64
    dropout_rate = 0.5
    n_hops = 2

    # 3. 开始训练
    tf.reset_default_graph()
    sess = tf.Session()
    model = QA(train_data.params['vocab_size'])
    sess.run(tf.global_variables_initializer())

    batching = (train_data.data['val']['inputs'].shape[0] // batch_size) * batch_size

    for i in range(epoch):
        total_cost, total_acc = 0, 0

        for k in range(0, batching, batch_size):
            # 问题　对话　对话长度
            batch_questions = train_data.data['val']['questions'][k: k + batch_size]
            batch_inputs = train_data.data['val']['inputs'][k: k + batch_size]
            batch_answers = train_data.data['val']['answers'][k: k + batch_size]

            batch_inputs_len = train_data.data['len']['inputs_len'][k: k + batch_size]
            batch_questions_len = train_data.data['len']['questions_len'][k: k + batch_size]
            batch_answers_len = train_data.data['len']['answers_len'][k: k + batch_size]
            acc, cost, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                    feed_dict={
                                        model.questions: batch_questions,
                                        model.inputs: batch_inputs,
                                        model.answers: batch_answers,
                                        model.questions_len: batch_questions_len,
                                        model.inputs_len: batch_inputs_len,
                                        model.answers_len: batch_answers_len,
                                        model.training: True
                                    })

            total_cost += cost
            total_acc += acc
            print('epoch: %d, step: %d, loss: %f, accuracy: %f'%(i, k // batch_size, cost, acc))

        total_cost /= (train_data.data['val']['inputs'].shape[0] // batch_size)
        total_acc /= (train_data.data['val']['inputs'].shape[0] // batch_size)
        print('epoch %d, avg cost %f, avg acc %f'%(i+1,total_cost,total_acc))

    testing_size = 32
    batch_questions = test_data.data['val']['questions'][:testing_size]
    batch_inputs = test_data.data['val']['inputs'][:testing_size]
    batch_inputs_len = test_data.data['len']['inputs_len'][:testing_size]
    batch_questions_len = test_data.data['len']['questions_len'][:testing_size]
    batch_answers_len = test_data.data['len']['answers_len'][:testing_size]
    batch_answers = test_data.data['val']['answers'][:testing_size]
    logits = sess.run(model.logits,
                      feed_dict={
                          model.questions: batch_questions,
                          model.inputs: batch_inputs,
                          model.inputs_len: batch_inputs_len,
                          model.questions_len: batch_questions_len,
                          model.answers_len: batch_answers_len,
                          model.training: False
                      })

    for i in range(testing_size):
        print('QUESTION:', ' '.join([train_data.vocab['idx2word'][k] for k in batch_questions[i]]))
        print('REAL:', train_data.vocab['idx2word'][batch_answers[i, 0]])
        print('PREDICT:', train_data.vocab['idx2word'][logits[i, 0]], '\n')
