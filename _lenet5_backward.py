# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import _lenet5_forward
import _lenet5_test

import numpy as np

BATCH_SIZE = 500
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.001
STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "/home/zhenxing/CYP450/CNN/CYP1A2/model_500"
MODEL_NAME = "CYP1A2_MODEL"


def trans_OH(label):
    newY = []
    for i in (label):
        if i == 0:
            newY.append([1, 0])
        else:
            newY.append([0, 1])
    NEWY = np.array(newY)
    return NEWY


def NextBatch(DATA_X, DATA_Y, BATCH_SIZE, i):
    BATCH_NUM = len(DATA_X) // BATCH_SIZE
    start = (i % BATCH_NUM) * BATCH_SIZE
    end = start + BATCH_SIZE
    if i % BATCH_NUM == 0:
        index = [k for k in range(len(DATA_X))]
        np.random.shuffle(index)
    reprentation = DATA_X[start:end]
    label = DATA_Y[start:end]
    return reprentation, label


def backward(DATA_X, DATA_Y,TDATA_X,TDATA_Y):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        _lenet5_forward.IMAGE_SIZE,
        _lenet5_forward.IMAGE_SIZE,
        _lenet5_forward.NUM_CHANNELS])

    y_label = tf.placeholder(tf.float32, [None, _lenet5_forward.OUTPUT_NODE])
    y_pre, y_pre_softmax = _lenet5_forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pre, labels=tf.argmax(y_label, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        len(DATA_X) // BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for n in range(10):
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            a1 = [i for i in range(len(DATA_X)) if (i - n) % 10 == 0]
            b1 = [i for i in range(len(DATA_X)) if not (i - n) % 10 == 0]
            trainCV_X = DATA_X[b1]
            trainCV_Y = DATA_Y[b1]
            val_CV_X = DATA_X[a1]
            val_CV_Y = DATA_Y[a1]

            for i in range(STEPS):

                xs, ys = NextBatch(trainCV_X, trainCV_Y, BATCH_SIZE, i)
                reshaped_xs = np.reshape(xs, (
                    BATCH_SIZE,
                    _lenet5_forward.IMAGE_SIZE,
                    _lenet5_forward.IMAGE_SIZE,
                    _lenet5_forward.NUM_CHANNELS))
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_label: ys})
                if (i+1) % 1000 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                if (i+1) % 1000 == 0:
                    print("After %d CV, %d training step(s), loss on training batch is %g." % (n+1, step, loss_value))
                    print("For trainset:")
                    _lenet5_test.test(val_CV_X, val_CV_Y)
                    print("For testset:")
                    _lenet5_test.test(TDATA_X, TDATA_Y)



