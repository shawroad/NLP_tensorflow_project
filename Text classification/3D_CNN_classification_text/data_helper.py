"""

@file  : data_helper.py

@author: xiaolu

@time  : 2019-12-19

"""
import os
import numpy as np
import random


def mkdir_if_not_exist(dirpath):

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    return dirpath


def read_and_clean_file(input_file, vocab):
    '''
    读取语料
    :param input_file:
    :param vocab:
    :return:
    '''
    lines = list(open(input_file, "r").readlines())
    # print(len(lines))  # 50000
    corpus_lines = []
    for line in lines:
        line_list = line.strip().split()
        corpus_lines.append(line_list)
    return corpus_lines


def load_data_and_labels(data_dir, vocab):
    '''
    加载数据以及标签
    :param data_dir: 数据所在位置
    :param vocab: 词表
    :return:
    '''
    datas = read_and_clean_file(data_dir+'train.datas', vocab)
    labels = [int(label_index) for label_index in open(data_dir+'train.labels', "r").readlines()]
    length = max(labels) + 1
    labels = np.eye(length)[labels]
    return [datas, labels]


def padding_sentences(input_sentences, padding_token, padding_sentence_length=None):
    #  如果你的长度没有给出我们按照最大的来,否则按照给出的来
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max(
        [len(sentence) for sentence in input_sentences])

    for sentence in input_sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] *
                            (max_sentence_length - len(sentence)))

    return input_sentences


def sentence2matrix(sentences, max_length, vocs):
    '''
    句子转为id序列
    :param sentences:
    :param max_length:
    :param vocs:
    :return:
    '''
    sentences_num = len(sentences)
    data_dict = np.zeros((sentences_num, max_length), dtype='int32')

    for index, sentence in enumerate(sentences):
        data_dict[index, :] = map2id(sentence, vocs, max_length)

    return data_dict


def map2id(sentence, voc, max_len):
    '''
    具体的一句话的id
    :param sentence:
    :param voc:
    :param max_len:
    :return:
    '''
    array_int = np.zeros((max_len,), dtype='int32')
    min_range = min(max_len, len(sentence))

    for i in range(min_range):
        item = sentence[i]
        array_int[i] = voc.get(item, voc['<UNK>'])

    return array_int


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    产生batch_size数据
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]

        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]


def load_testfile_and_labels(input_text_file, input_label_file, vocab, num_samples=-1):
    x_text = read_and_clean_file(input_text_file, vocab)

    y = None if not os.path.exists(input_label_file) else map(
        int, list(open(input_label_file, "r").readlines()))
    y = [i for i in y]

    # get some samples randomly form testfile, -1 means all samples
    if num_samples != -1:
        index_list = [i for i in range(len(x_text))]
        samples_index = random.sample(index_list, num_samples)
        samples = []
        labels = []
        for index in samples_index:
            samples.append(x_text[index])
            labels.append(y[index])
        x_text = samples
        y = labels
    return (x_text, y)


# if __name__ == '__main__':
#     # data_dir = './data/train_data/'
#     # huhu = load_data_and_labels(data_dir, './runs/vocab')
#     # print(huhu)
#     input_label_file = './data/test_data/test.labels'
#     y = map(int, list(open(input_label_file, "r").readlines()))
#     y = [i for i in y]


