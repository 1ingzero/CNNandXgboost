# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import tensorflow as tf
import _lenet5_forward
import  _lenet5_backward
import numpy as np
from sklearn import metrics
import _lenet5_backward
# 定义了一个函数来求解SP和SE


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




TEST_INTERVAL_SECS = 50
def test (DATA_X, DATA_Y):
    with tf.Graph().as_default() as g:

        x = tf.placeholder(tf.float32,[
            len(DATA_X),
            _lenet5_forward.IMAGE_SIZE,
            _lenet5_forward.IMAGE_SIZE,
            _lenet5_forward.NUM_CHANNELS])
        y_label = tf.placeholder(tf.float32, [None, _lenet5_forward.OUTPUT_NODE])
        y_pre, y_softmax_pre = _lenet5_forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        while True:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                ckpt = tf.train.get_checkpoint_state(_lenet5_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(DATA_X,(
                        len(DATA_X),
                        _lenet5_forward.IMAGE_SIZE,
                        _lenet5_forward.IMAGE_SIZE,
                        _lenet5_forward.NUM_CHANNELS))
                    Y_pre, Y_label, Y_pre_softmax = sess.run([y_pre,y_label,y_softmax_pre], feed_dict={x: reshaped_x, y_label:DATA_Y})
                    Prediction = tf.cast(tf.argmax(Y_pre_softmax, 1),dtype=np.float32)
                    AUC = metrics.roc_auc_score(Y_label[:, 1], Y_pre_softmax[:, 1])
                    print("After %s training step(s)："%(global_step))
                    SPSEQMCC(Y_label[:, 1], Prediction.eval())
                    print("AUC:", AUC)

                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main ():
    _X = "/home/zhenxing/CYP450/CNN/CYP2C19/data/CYP2C19_test_x.txt"
    _Y = "/home/zhenxing/CYP450/CNN/CYP2C19/data/test_Y.txt"
    DATA_X = np.loadtxt(_X, dtype=np.float32)
    DATA_Y = np.loadtxt(_Y, dtype=np.float32)
    DATA_Y_OH = _lenet5_backward.trans_OH(DATA_Y)
    test(DATA_X, DATA_Y_OH)

if __name__ == '__main__':
    main()