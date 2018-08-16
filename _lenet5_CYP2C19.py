# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import _lenet5_forward
import _lenet5_backward
import _lenet5_test

if _lenet5_backward