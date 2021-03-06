#导入必要的库
import sys
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

#定义'max_depth'和'min_child_weight'的寻优函数
if __name__=='__main__':
    train = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem883/CYP2C9_PaDEL12_pcfp_train.csv')
    test = pd.read_csv('/home/zhenxing/DATA/CYP DATASET/Pubchem883/CYP2C9_PaDEL12_pcfp_test.csv')
    target = 'label'
    IDcol = 'Name'
    predictors = [x for x in train.columns if x not in [target, IDcol]]

    param_test2 = {
    'max_depth':list(range(3,10,1)),
    'min_child_weight':list(range(1,6,1))}

    gsearch2 = GridSearchCV( estimator = XGBClassifier( learning_rate =0.1, n_estimators=445, max_depth=5,
                         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                         objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                         param_grid = param_test2,
                         scoring='roc_auc',
                         n_jobs=4,
                         iid=False,
                         cv=5)
    gsearch2.fit(train[predictors],train[target])
    print_best_score(gsearch2,param_test2)
    sys.exit()