import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import  metrics
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
Figure1_title ='The ROC curve of CYP3A4 Training set'
Figure2_title ='The ROC curve of CYP3A4 test set'
Figure3_title ='Feature Importances Of CYP3A4'
Trainset = '/home/zhenxing/DATA/CYP DATASET/Pubchem884/CYP3A4_PaDEL12_pcfp_train.csv'
Testset = '/home/zhenxing/DATA/CYP DATASET/Pubchem884/CYP3A4_PaDEL12_pcfp_test.csv'



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
    print("TP,FN,TN,FP:", TP, FN, TN, FP)
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
def modelfit(alg, dtrain,dtest):
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

    #计算ROC曲线
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(dtrain[target], dtrain_predprob)
    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(dtest[target], dtest_predprob)

    plt.figure(1)
    plt.plot(fpr_train, tpr_train)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(Figure1_title)
    plt.show()

    plt.figure(2)
    plt.plot(fpr_test, tpr_test)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(Figure2_title)
    plt.show()

    plt.figure(3)
    rcParams['figure.figsize'] = 12, 4
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp_50 = feat_imp[:50]
    feat_imp_50.plot(kind='bar', title=Figure3_title)
    plt.ylabel('Feature Importance Score')
    plt.show()

#读入数据
train = pd.read_csv(Trainset)
test = pd.read_csv(Testset)
target = 'label'
IDcol = 'Name'
predictors = [x for x in train.columns if x not in [target, IDcol]]

#搭建模型
xgb1 = XGBClassifier(
    learning_rate=0.01,
    n_estimators=5000,
    max_depth=5,
    min_child_weight=3,
    gamma=0.23,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=5e-6,
    reg_lambda=1,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)
modelfit(xgb1, train, test)