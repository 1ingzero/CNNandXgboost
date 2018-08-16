# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import _lenet5_forward
import  _lenet5_backward
import numpy as np
from sklearn import metrics
# 定义了一个函数来求解SP和SE
BATCH_SIZE = 593
NUM = 16

def SPSEQMCC(label, prediction):
    TP, FN, TN, FP = 0, 0, 0, 0
    for i in range(len(label)):
        if label[i] == 0:
            if prediction[i] == 0:
                TN = TN + 1
            else:
                FP = FP + 1
        else:
            if prediction[i] == 0:
                FN = FN + 1
            else:
                TP = TP + 1
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    print("ACC：", ACC)
    print("SE：", SE)
    print("SP：", SP)
    L = (TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)
    if L == 0:
        print("No MCC")
    else:
        MCC = (TP * TN - FP * FN) / (((TN + FN) * (TN + FP) * (TP + FN) * (TP + FP)) ** 0.5)
        print("MCC:", MCC)


def test (DATA_X, DATA_Y):
    with tf.Graph().as_default() as g:

        x = tf.placeholder(tf.float32,[
            BATCH_SIZE,
            _lenet5_forward.IMAGE_SIZE,
            _lenet5_forward.IMAGE_SIZE,
            _lenet5_forward.NUM_CHANNELS])
        y_label = tf.placeholder(tf.float32, [None, _lenet5_forward.OUTPUT_NODE])
        y_pre, y_softmax_pre = _lenet5_forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)


        with tf.Session( ) as sess:
            ckpt = tf.train.get_checkpoint_state(_lenet5_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                reshaped_x = np.reshape(DATA_X,(
                        len(DATA_X),
                        _lenet5_forward.IMAGE_SIZE,
                        _lenet5_forward.IMAGE_SIZE,
                        _lenet5_forward.NUM_CHANNELS))
                #创建一个数组便于相加
                Y_pre1 = np.zeros(shape=(NUM, 2))
                Y_label1 = np.zeros(shape=(NUM, 2))
                Y_pre_softmax1 = np.zeros(shape=(NUM, 2))
                for i in range(NUM):
                    DATA_X_batch = reshaped_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    DATA_Y_batch = DATA_Y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                    Y_pre,Y_label,Y_pre_softmax = sess.run([y_pre,y_label,y_softmax_pre], feed_dict={x: DATA_X_batch, y_label:DATA_Y_batch})
                    Y_pre1 = np.concatenate((Y_pre1, Y_pre), axis=0)
                    Y_label1 = np.concatenate((Y_label1, Y_pre), axis=0)
                    Y_pre_softmax1 = np.concatenate((Y_pre_softmax1, Y_pre), axis=0)
                    Y_pre2 =Y_pre1[1 * BATCH_SIZE:NUM * BATCH_SIZE]
                    Y_label2=Y_label[1 * BATCH_SIZE:NUM * BATCH_SIZE]
                    Y_pre_softmax2 =Y_pre_softmax1[1 * BATCH_SIZE:NUM * BATCH_SIZE]

                Prediction = tf.cast(tf.argmax(Y_pre_softmax2, 1),dtype=np.float32)
                AUC = metrics.roc_auc_score(Y_label2[:, 1], Y_pre_softmax2[:, 1])
                SPSEQMCC(Y_label2[:, 1], Prediction.eval())
                print("AUC:", AUC)
            else:
                print('No checkpoint file found')