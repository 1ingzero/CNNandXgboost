#导入必要的库
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV

#定义输出寻优结果函数
def print_best_score(gsearch, param_test):
    # 输出best score
    print("grid_scores:",gsearch.grid_scores_)
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


#定义'subsample'和'colsample_bytree'的寻优函数
if __name__=='__main__':
    train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem899/CYP2C19_PaDEL12_pcfp_train.csv')
    test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem899/CYP2C19_PaDEL12_pcfp_test.csv')
    target = 'label'
    IDcol = 'Name'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    param_test4 = { 'subsample': list([i / 10.0 for i in range(6, 10)]),
                    'colsample_bytree': list([i / 10.0 for i in range(6, 10)]) }

    gsearch4 = GridSearchCV( estimator = XGBClassifier( learning_rate =0.1, n_estimators=500, max_depth=7,
                         min_child_weight=5, gamma=0,  colsample_bytree=0.8,subsample=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                         param_grid = param_test4,
                         scoring='roc_auc',
                         n_jobs=4,
                         iid=False,
                         cv=5)
    gsearch4.fit(train[predictors],train[target])
    print_best_score(gsearch4,param_test4)