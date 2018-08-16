#导入必要的库
import sys
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import  metrics
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4



#定义评估系数计算函数
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



#定义构建模型的函数
def modelfit(alg, dtrain,dtest, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print(cvresult[len(cvresult)-1:len(cvresult)])
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Predict test set:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]


    # Print model report:
    print ("\nModel Report (Train):")
    SPSEQMCC(dtrain[target].values, dtrain_predictions)
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    print("\nModel Report (Test):")
    SPSEQMCC(dtest[target].values, dtest_predictions)
    print("AUC Score (Test): %f" % metrics.roc_auc_score(dtest[target], dtest_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp_50 = feat_imp[:50]
    feat_imp_50.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

#读入数据
train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem891/CYP2D6_PaDEL12_pcfp_train.csv')
test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem891/CYP2D6_PaDEL12_pcfp_test.csv')
target = 'label'
IDcol = 'Name'
predictors = [x for x in train.columns if x not in [target, IDcol]]

#搭建模型
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=2000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train,test, predictors)
sys.exit()
