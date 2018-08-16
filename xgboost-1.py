# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import xgboost as xgb
import pandas as pd
# 计算分类正确率
from sklearn.metrics import accuracy_score
#from sklearn.metrics import roc_auc_score

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

# read in data，数据在xgboost安装的路径下的demo目录,现在我们将其copy到当前代码下的data目录
train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem884/CYP3A4_PaDEL12_pcfp_train.csv')
test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem884/CYP3A4_PaDEL12_pcfp_test.csv')
target = 'label'
IDcol = 'Name'

predictors = [x for x in train.columns if x not in [target, IDcol]]
dtrain = xgb.DMatrix(train[predictors], label=train[target])
dtest = xgb.DMatrix(test[predictors], label=test[target])
# specify parameters via map
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
print(param)

# 设置boosting迭代计算次数
num_round = 5000
bst = xgb.train(param, dtrain, num_round)  # dtrain是训练数据集


train_preds = bst.predict(dtrain)
#train_predict_proba = bst.predict_proba(dtrain)[:, 1]
train_predictions = [round(value) for value in train_preds]
y_train = dtrain.get_label()
train_accuracy = accuracy_score(y_train, train_predictions);iouhjkhkjhjjjjjhjkhjkhhjkjkjjkjkjkhkjbkjjjjjjjjjjjjjhjh
#train_AUC = roc_auc_score(y_train,train_predict_proba)
print("Train Accuary: %.2f%%" % (train_accuracy * 100.0))
SPSEQMCC(y_train, train_predictions)
#print("Train AUC: %.2f%%" % (train_AUC * 100.0))

# make prediction
preds = bst.predict(dtest)
predictions = [round(value) for value in preds]
#test_predict_proba = bst.predict_proba(dtest)[:, 1]

y_test = dtest.get_label()
test_accuracy = accuracy_score(y_test, predictions)
#test_AUC = roc_auc_score(y_test,test_predict_proba)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
SPSEQMCC(y_test, predictions)
#print("Test AUC: %.2f%%" % (test_AUC * 100.0))
