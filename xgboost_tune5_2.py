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

#定义'reg_alpha'的寻优函数
if __name__=='__main__':
    train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem891/CYP2D6_PaDEL12_pcfp_train.csv')
    test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem891/CYP2D6_PaDEL12_pcfp_test.csv')
    target = 'label'
    IDcol = 'Name'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    param_test5_2= {   'reg_alpha':list([ 5e-6, 1e-5,5e-5]),
                       'reg_lambda':list([0.5,1,5])}

    gsearch5_2 = GridSearchCV( estimator = XGBClassifier( learning_rate =0.1, n_estimators=585, max_depth=5,
                         min_child_weight=3, gamma=0.23, colsample_bytree=0.8, subsample=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                         param_grid = param_test5_2,
                         scoring='roc_auc',
                         n_jobs=4,
                         iid=False,
                         cv=5)
    gsearch5_2.fit(train[predictors],train[target])
    print_best_score(gsearch5_2,param_test5_2)