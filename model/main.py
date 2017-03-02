import pickle
import random
import numpy as np
import math
import time
import os

import tensorflow as tf
from tensorflow.contrib import rnn

class BaseConfig(object):
    def __init__(self):
        self.feature_size = 53
        self.predict_day = 14
        self.epoch = 100
        self.batch_size = 50
        self.step = 50
        self.rnn_layer_nums = 2
        self.hidden_size = 600
        self.class_size = 2000
        self.keep_prob = 1.0 #0.85
        self.lr_decay = 0.95
        self.learning_rate = 0.5
        self.optimizer = "nesterov"

class BaseReader(object):
    def __init__(self, data_path, batch_size, step):
        self.predict_day = 14
        self.batch_size = batch_size
        self.step = step
        self.total_data_num = 0
        self.max_class_size = 0
        shop_feature = pickle.load(open(data_path, "rb"))

        # generate input data: ([step, F], scalar)
        self.input_data = []
        shop_num, T, F = np.shape(shop_feature)[:]
        for shop_id in range(shop_num):
            for t in range(0, T-self.step-self.predict_day):
                t_pred_s = t + self.step
                t_pred_e = t + self.step + self.predict_day
                inputs = \
                    shop_feature[shop_id][t : t_pred_s][:]
                labels = []
                for tmp_i in range(t_pred_s, t_pred_e):
                    tmp = 0
                    for tmp_j in range(25):
                        tmp += shop_feature[shop_id][tmp_i][tmp_j]
                    if tmp > self.max_class_size:
                        self.max_class_size = tmp
                    labels.append(tmp)
                self.input_data.append((inputs, labels))
        self.total_data_num = len(self.input_data)
        print("total data:", self.total_data_num)
        print("max class size", self.max_class_size)

    def generator(self):
        random.shuffle(self.input_data)
        train_data = self.input_data[:self.total_data_num * 9 // 10]
        valid_data = self.input_data[self.total_data_num * 9 // 10: ]

        def _batch_generator(data):
            for i in range(len(data) // self.batch_size):
                batch_data = [
                    self.input_data[bi][0] for bi in 
                    range(i*self.batch_size, (i+1)*self.batch_size)
                ]
                batch_label = [
                    self.input_data[bi][1] for bi in
                    range(i*self.batch_size, (i+1)*self.batch_size)
                ]
                yield (batch_data, batch_label)

        return _batch_generator(train_data), _batch_generator(valid_data)

class Model(object):
    def __init__(self, config, is_train):
        # input placeholder
        self.inputs = inputs = tf.placeholder(
            tf.float32, [config.batch_size, config.step, config.feature_size]
        )  # [b, t, f]
        self.labels = labels = tf.placeholder(
            tf.int32, [config.batch_size, config.predict_day]
        )  # [b, 14]

        # learning method
        with tf.variable_scope('global_step'):
            self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope('lr'):
            initial_lr = tf.Variable(
                config.learning_rate, trainable=False
            )
            self.lr = tf.train.exponential_decay(
                initial_lr, self.global_step,
                config.decay_steps, config.lr_decay,
                staircase=True
            )

        # optimizer
        if is_train:
            if config.optimizer == "nesterov":
                optimizer = tf.train.MomentumOptimizer(
                    self.lr, 0.9, use_nesterov=True
                )
            elif config.optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise ValueError("Unknowned optimizer %s" % config.optimizer)

        # input dropout
        if config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # memory rnn layer
        # [b, t, f] -> [b, t, h]
        memory_rnn_cells = []
        xavier = tf.contrib.layers.xavier_initializer(uniform=False)
        for i in range(config.rnn_layer_nums):
            with tf.variable_scope('memory_rnn_cell%d' % i,
                                   initializer=xavier):
                lstm_cell = \
                    tf.contrib.rnn.BasicLSTMCell(config.hidden_size)
                if config.keep_prob < 1:
                    lstm_cell = rnn.DropoutWrapper(
                        lstm_cell, output_keep_prob=config.keep_prob
                    )
                memory_rnn_cells.append(lstm_cell)
        memory_rnn_cells = rnn.MultiRNNCell(memory_rnn_cells)
        memory_outputs, memory_states = tf.nn.dynamic_rnn(
            memory_rnn_cells, inputs, dtype=tf.float32,
            scope='memory_rnn'
        )

        # predict process (step by step)

#        predict_rnn_cells = rnn.MultiRNNCell(
#            [tf.contrib.rnn.BasicLSTMCell(config.hidden_size)] *\
#                config.rnn_layer_nums
#        )
#        predict_result = []

        last_output = memory_outputs[:][-1][:]
        last_output = tf.reshape(
            last_output,
            [config.batch_size, config.hidden_size]
        )  # [b, h]
        with tf.variable_scope("softmax"):
            softmax_w = tf.get_variable(
                "softmax_w",
                [config.hidden_size, config.class_size]
            )
            softmax_b = tf.get_variable(
                "softmax_b",
                [config.class_size]
            )
#        logit = tf.matmul(last_output, softmax_w) + softmax_b
#        # [b, h] * [h, v] -> [b, v]
#        predict_start_input = tf.cast(
#            tf.reshape(
#                tf.argmax(logit, 1), [config.batch_size, 1]),
#            tf.float32
#        )
#        # [b, v] -> [b, 1]
#        print("predict_start", predict_start_input)

        logits = []
        predictions = []
        for t in range(config.predict_day):
            with tf.variable_scope(
                tf.get_variable_scope(),
                # reuse=True if t == 0 else None):
                reuse=None):
                if t == 0:
                    (output, state) = memory_rnn_cells(
                        last_output, memory_states
                    )
                else:
                    tf.get_variable_scope().reuse_variables()
                    (output, state) = memory_rnn_cells(
                        output, state
                    )
                logit = tf.matmul(output, softmax_w) + softmax_b
                # [b, h] -> [b, v]
                logits.append(logit) # t * [b, v]
                predictions.append(tf.reshape(
                    tf.argmax(logit, 1), [config.batch_size, 1])
                )

        # calculate loss
        logits = tf.reshape(
            tf.concat(logits, 1), [-1, config.class_size])
        targets = tf.reshape(self.labels, [-1]) # [b, t] -> [b * t]
        weights = tf.cast(tf.ones_like(targets), tf.float32)
        self.cost = cost = \
                tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [targets],
            [weights]
        )
        self.predictions = tf.concat(predictions, 1)
#        self.cost = cost = tf.reduce_sum(
#            tf.abs(tf.truediv(
#                tf.subtract(predict_result, self.labels),
#                tf.add(predict_result, self.labels)))
#        )
        if is_train:
            grads_and_vars = optimizer.compute_gradients(
                cost, tf.trainable_variables()
            )
            self.train_op = optimizer.apply_gradients(
                grads_and_vars, self.global_step
            )

        print("Model build finish!")

if __name__ == "__main__":
    data_path = "./pkl_files/shop_feature.pkl"
    model_path = None
    work_path = "./saved_model/0226"
    config = BaseConfig()
    print("loading data...")
    reader = BaseReader(data_path, config.batch_size, config.step)
    print("loading done!")
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        # calculate decay_steps and step_per_valid
        config.class_size = reader.max_class_size + 1
        config.decay_steps = \
            (2 * reader.total_data_num // config.batch_size)
        step_per_valid = \
            reader.total_data_num // config.batch_size // 10
        step_per_print_log = \
            reader.total_data_num // config.batch_size // 100

#        config.decay_steps = 60000

        # print config log
        print("/****config****/")
        print("model config is:", config.__dict__)
        print("/*************/")

        # build model
        with tf.variable_scope("model"):
            train_model = Model(config, is_train=True)

        config.keep_prob = 1.0
        with tf.variable_scope("model", reuse=True):
            valid_model = Model(config, is_train=False)

        # init variables
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        if model_path is None:
            sess.run(tf.global_variables_initializer())
            print("model init")
        else:
            saver.restore(sess, save_path=model_path)
            print("model restored from", model_path)

        # run
        min_cost = 0.0
        pre_time = time.time()
        step = 0
        for e in range(config.epoch):
            train_data_iter, valid_data_iter = reader.generator()
            for (b_input, b_label) in train_data_iter:
                train_cost, _, lr, step, labels, res = sess.run(
                    [train_model.cost,
                     train_model.train_op,
                     train_model.lr,
                     train_model.global_step,
                     train_model.labels,
                     train_model.predictions
                    ],
                    {train_model.inputs: b_input,
                     train_model.labels: b_label
                    }
                )
                labels = np.array(labels).astype(float)
                res = np.array(res).astype(float) + 1e-5
                loss = np.sum(np.abs(
                    np.divide(np.subtract(labels, res), np.add(labels, res))
                )) / config.batch_size / config.predict_day
                print("loss", loss)

                # print log
                if step % step_per_print_log == 0:
                    saver.save(sess, os.path.join(work_path, 'latest'))
                    print("step", step)
                    print("cost", train_cost)
                    print("loss", loss)
                    print("labels", labels)
                    print("results", res)
                    print("passed time", time.time() - pre_time)
                    pre_time = time.time()
                if step % step_per_valid == 0:
#                if True:
                    print("---run validation---")
                    vlosses = []
                    for (bv_input, bv_label) in valid_data_iter:
                        labels, res = sess.run(
                            [valid_model.labels,
                             valid_model.predictions
                            ],
                            {valid_model.inputs: bv_input,
                             valid_model.labels: bv_label
                            }
                        )
                        labels = np.array(labels).astype(float)
                        res = np.array(res).astype(float) + 1e-5
                        loss = np.sum(np.abs(
                            np.divide(np.subtract(labels, res),
                                      np.add(labels, res))
                        )) / config.batch_size / config.predict_day
                        print("vloss", loss)
                        vlosses.append(loss)
                    print("mean cost", np.mean(vlosses))
                    if np.mean(vlosses) < min_cost:
                        print("* best model *")
                        saver.save(sess, os.path.join(work_path, 'best'))
                        min_cost = np.mean(vlosses)
                        print("\n")
