"""

@file  : train.py

@author: xiaolu

@time  : 2019-12-19

"""
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import datetime
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, _log_fg_cy, _log_black, rainbow
import data_helper
from model import TextCNN
import codecs

# Parameters
# Data loading params
tf.flags.DEFINE_float("dev_per", 0.0001,
                      "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("input_test_file", "./data/test_data/test.datas",
                       "Data source for the test_data data.")
tf.flags.DEFINE_string("input_label_file", "./data/test_data/test.labels",
                       "Label file for test_data text data source.")

# Model Hyperparameters
# n_gram, pixel_weight, pixel_height, num_filters,
tf.flags.DEFINE_integer("n_gram", 3,
                        "num_filters (default: 3)")
tf.flags.DEFINE_integer("pixel_weight", 131,
                        "Dimensionality of character embedding (default: 20)")
tf.flags.DEFINE_integer("pixel_height", 20,
                        "Dimensionality of character embedding (default: 131)")
tf.flags.DEFINE_integer("num_filters", 50,
                        "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1e-4,
                      "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("max_sentence_len", 200,
                        "max length of sentences (default: 64)")

# Training parameters
tf.flags.DEFINE_float("learning_rate", 1e-4,
                      "learning rate (default:1e-4)")
tf.flags.DEFINE_integer("batch_size", 64,
                        "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30,
                        "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("test_every", 1000,
                        "Test model on test_data set after this many steps(default:500)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")
tf.flags.DEFINE_string("padding_token", '<PAD>',
                       "uniform sentences")

FLAGS = tf.flags.FLAGS

_log_warning("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    _log_info("{}={}".format(attr.upper(), value.value))
_log_info("")


# Output directory for models and summaries
out_dir = data_helper.mkdir_if_not_exist("./runs")

# Load data
_log_warning("Loading data...")


# 建立词表
vocab_tokens = [line.strip() for line in codecs.open('./runs/vocab', 'r', 'utf-8').readlines()]
vocsize = len(vocab_tokens)
vocab = {}
for (i, token) in enumerate(vocab_tokens):
    vocab[token] = i

# 加载的是训练集数据及标签
x_text, y = data_helper.load_data_and_labels('./data/train_data/', './runs/vocab')

# 进行padding  传入的是语料, 填充的标志, 最大填充的长度
sentences = data_helper.padding_sentences(
    x_text, FLAGS.padding_token, FLAGS.max_sentence_len)

print("len(x_text)", len(x_text))
print("len(y)", len(y))
# Build vocabulary

# 将语料转为对应的id
x = np.array(data_helper.sentence2matrix(sentences, FLAGS.max_sentence_len, vocab))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))

x_train = x[shuffle_indices]
y_train = y[shuffle_indices]
# print(x_train.shape)  # (50000, 200)
# print(y_train.shape)  # (50000, 5)


data_len = len(x_train)
_log_info("Total length: {:d}".format(data_len))

# Training
global_graph = tf.Graph()

with global_graph.as_default():
    sess = tf.Session(graph=global_graph,
                      config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    with sess.as_default():
        model = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=vocsize,
            batch_size=FLAGS.batch_size,
            n_gram=FLAGS.n_gram,
            pixel_weight=FLAGS.pixel_weight,
            pixel_height=FLAGS.pixel_height,
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        dlearning_rate = tf.constant(FLAGS.learning_rate)

        optimizer = tf.train.AdamOptimizer(dlearning_rate)

        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        loss_summary = tf.summary.scalar('loss', model.loss)
        acc_summary = tf.summary.scalar('accuracy', model.accuracy)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")

        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "model"))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        checkpoint_file = tf.train.latest_checkpoint('./runs/model')

        if checkpoint_file != None:
            saver.restore(sess, checkpoint_file)
            _log_info("restore session from checkpoint files")
        else:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, lr, loss, l2_loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, dlearning_rate, model.loss, model.l2_losses, model.accuracy],
                feed_dict
            )

            time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")
            rainbow("train set:*** {}: step {}, learning_rate {:5f}, loss {:g}, l2_loss {:g}, acc {:g}".format(
                time_str, step, lr, loss, l2_loss, accuracy), time_tag=True)

            train_summary_writer.add_summary(summaries, step)

            return loss

        batches = data_helper.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:

            x_batch, y_batch = zip(*batch)
            train_loss = train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.test_every == 0:
                _log_warning("Loading  Test data...")

                x_raw, y_data = data_helper.load_testfile_and_labels(
                    FLAGS.input_test_file, FLAGS.input_label_file, vocab, num_samples=256)

                sentences = data_helper.padding_sentences(
                    x_raw, FLAGS.padding_token, FLAGS.max_sentence_len)

                x_test = np.array(data_helper.sentence2matrix(
                    sentences, FLAGS.max_sentence_len, vocab))

                y_test = []

                for item in y_data:
                    label = np.zeros([y_train.shape[1]])
                    label[item] = 1
                    y_test.append(label)

                y_test = tuple(y_test)

                step, loss, accuracy = sess.run([global_step, model.loss, model.accuracy],
                                                {
                                                    model.input_x: x_test,
                                                    model.input_y: y_test,
                                                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                                                })
                time_str = datetime.datetime.now().isoformat()
                log_str = "Time::{}, Step::{}, Loss::{}, Accuracy::{} on Test data.".format(
                    time_str, current_step, loss, accuracy)
                _log_info(log_str)

                with open(out_dir+'/training_log.txt', 'a') as out_put_file:
                    out_put_file.write(log_str + '\n')

                tf.train.write_graph(
                    sess.graph_def, checkpoint_dir, 'classify_text.pbtxt')

                saver.save(sess, checkpoint_dir +
                           '/classify_text.ckpt', global_step=step)

                _log_info("Accuray ::{},Save model checkpoint to {}\n".format(
                    accuracy, checkpoint_dir))
