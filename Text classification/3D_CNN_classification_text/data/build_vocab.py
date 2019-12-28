"""

@file  : build_vocab.py

@author: xiaolu

@time  : 2019-12-19

"""
from collections import Counter


def load_data(path):
    '''
    加载数据
    :param path:
    :return:
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            data.append(line)
    return data


def vocab_gen(total_data):
    '''
    生成词表
    :param total_data:
    :return:
    '''
    total_str = ' '.join(total_data)
    # 分词
    words = total_str.split()
    vocab_size = len(list(set(words)))

    counter = Counter(words).most_common(vocab_size)

    vocab = []
    vocab.append('<UNK>')
    for w, _ in counter:
        vocab.append(w)
    vocab.append('<PAD>')

    string = '\n'.join(vocab)
    with open('../runs/vocab', 'w') as f:
        f.write(string)


if __name__ == '__main__':
    train_path = './train_data/train.datas'
    test_path = './test_data/test.datas'

    train_data = load_data(train_path)
    test_data = load_data(test_path)

    total_data = []
    total_data.extend(train_data)
    total_data.extend(test_data)

    vocab_gen(total_data)