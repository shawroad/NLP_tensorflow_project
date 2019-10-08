"""

@file  : 6-bert.py

@author: xiaolu

@time  : 2019-09-26

"""
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import time


class Model:
    def __init__(self, dimension_output, learning_rate=2e-5):
        '''
        :param dimension_output:
        :param learning_rate:
        '''
        self.X = tf.placeholder(tf.int32, [None, None])
        self.segment_ids = tf.placeholder(tf.int32, [None, None])
        self.input_masks = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])

        # 实例化bert模型
        model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.X,
            input_mask=self.input_masks,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )

        # 获取模型的输出 进行dense
        output_layer = model.get_pooled_output()
        self.logits = tf.layers.dense(output_layer, dimension_output)
        self.logits = tf.identity(self.logits, name='logits')

        # 损失
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        # 使用bert内置的优化器
        self.optimizer = optimization.create_optimizer(self.cost, learning_rate, num_train_steps, num_warmup_steps, False)

        # 算准确率
        correct_pred = tf.equal(tf.argmax(self.logits, 1, output_type=tf.int32), self.Y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    '''
    把两句话的长度之和缩减到不超过maxlen-3
    :param tokens_a: 第一句话
    :param tokens_b: 第二句话
    :param max_length: 最大长度
    :return:
    '''
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


if __name__ == '__main__':
    # 词表　与训练权重　配置文件
    BERT_VOCAB = 'uncased_L-12_H-768_A-12/vocab.txt'
    BERT_INIT_CHKPNT = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
    BERT_CONFIG = 'uncased_L-12_H-768_A-12/bert_config.json'

    tokenization.validate_case_matches_checkpoint(True, '')
    tokenizer = tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)

    MAX_SEQ_LENGTH = 100

    df = pd.read_csv('./data/quora_duplicate_questions.tsv', delimiter='\t').dropna()
    print(df.head())

    left, right, label = df['question1'].tolist(), df['question2'].tolist(), df['is_duplicate'].tolist()

    input_ids, input_masks, segment_ids = [], [], []

    for i in range(len(left)):
        # 分词
        tokens_a = tokenizer.tokenize(left[i])
        tokens_b = tokenizer.tokenize(right[i])

        # 将两句话长度缩减到不超过100
        _truncate_seq_pair(tokens_a, tokens_b, MAX_SEQ_LENGTH - 3)

        tokens = []
        segment_id = []

        tokens.append("[CLS]")
        segment_id.append(0)

        # 第一句话全部标记为0
        for token in tokens_a:
            tokens.append(token)
            segment_id.append(0)

        tokens.append("[SEP]")
        segment_id.append(0)

        # 第二句话全部标记为1
        for token in tokens_b:
            tokens.append(token)
            segment_id.append(1)

        tokens.append("[SEP]")
        segment_id.append(1)

        input_id = tokenizer.convert_tokens_to_ids(tokens)   # 将文本转为id序列
        input_mask = [1] * len(input_id)   # 制造mask  有文本就是1,否则就是0

        while len(input_id) < MAX_SEQ_LENGTH:
            input_id.append(0)
            input_mask.append(0)
            segment_id.append(0)

        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)

    # 配置文件
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)

    epoch = 10
    batch_size = 60
    warmup_proportion = 0.1
    num_train_steps = int(len(left) / batch_size * epoch)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    dimension_output = 2
    learning_rate = 1e-5

    tf.reset_default_graph()
    sess = tf.Session()
    model = Model(dimension_output, learning_rate)

    sess.run(tf.global_variables_initializer())

    var_lists = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='bert')
    saver = tf.train.Saver(var_list=var_lists)

    saver.restore(sess, BERT_INIT_CHKPNT)

    train_input_ids, test_input_ids, train_input_masks, test_input_masks, train_segment_ids, test_segment_ids, train_Y, test_Y = train_test_split(
        input_ids, input_masks, segment_ids, label, test_size=0.2
    )

    EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0

    while True:

        if CURRENT_CHECKPOINT == EARLY_STOPPING:
            print('break epoch:%d\n' % EPOCH)
            break

        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0

        # 训练
        for i in range(0, len(train_input_ids), batch_size):
            index = min(i + batch_size, len(train_input_ids))
            batch_x = train_input_ids[i: index]
            batch_masks = train_input_masks[i: index]
            batch_segment = train_segment_ids[i: index]
            batch_y = train_Y[i: index]

            acc, cost, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                    feed_dict={
                                        model.Y: batch_y,
                                        model.X: batch_x,
                                        model.segment_ids: batch_segment,
                                        model.input_masks: batch_masks
                                    },
                                    )
            assert not np.isnan(cost)
            train_loss += cost
            train_acc += acc
            print('training--epoch: %d, step: %d, loss: %f, accuracy: %f' % (EPOCH, i // batch_size, cost, acc))

        # 测试
        for i in range(0, len(test_input_ids), batch_size):
            index = min(i + batch_size, len(test_input_ids))
            batch_x = test_input_ids[i: index]
            batch_masks = test_input_masks[i: index]
            batch_segment = test_segment_ids[i: index]
            batch_y = test_Y[i: index]
            acc, cost = sess.run([model.accuracy, model.cost],
                                 feed_dict={
                                     model.Y: batch_y,
                                     model.X: batch_x,
                                     model.segment_ids: batch_segment,
                                     model.input_masks: batch_masks
                                 })

            test_loss += cost
            test_acc += acc
            print('testing--epoch: %d, step: %d, loss: %f, accuracy: %f' % (EPOCH, i // batch_size, cost, acc))

        train_loss /= len(train_input_ids) / batch_size
        train_acc /= len(train_input_ids) / batch_size
        test_loss /= len(test_input_ids) / batch_size
        test_acc /= len(test_input_ids) / batch_size

        if test_acc > CURRENT_ACC:
            print('epoch: %d, pass acc: %f, current acc: %f' % (EPOCH, CURRENT_ACC, test_acc))
            CURRENT_ACC = test_acc
            CURRENT_CHECKPOINT = 0
        else:
            CURRENT_CHECKPOINT += 1

        print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n' % (EPOCH, train_loss, train_acc, test_loss, test_acc))
        EPOCH += 1

