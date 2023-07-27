##################
# hyperopt参数搜索
##################
import pickle
import pandas as pd
import xgboost as xgb
from os.path import join
import numpy as np

from utils import Preprocess
from utils import XgbCustomEval
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, f1_score)
from tqdm import tqdm
from hyperopt import fmin, hp, Trials, space_eval, tpe

PATH_DATA_ROOT = '../baseline_data/out/'
PATH_DATA_TRAIN = 'pp_train_no_0.csv'
PATH_DATA_TEST = 'pp_test.csv'
SKF_SPLIT = 5
RANDOM_STATE = 2023
ADDITIONAL_FEATURES = [
    'para_len', 'num_eng', 'num_num', 'num_chn',            # 数值特征
    'percent_chn', 'percent_eng', 'percent_num',            # 数值特征
    'kw_hongguan', 'kw_celue', 'kw_hangye',                 # 研报类别关键词
    'kw_gongsi', 'kw_jijin', 'kw_gegu',                     # 研报类别关键词
    'kw_yeji', 'has_fenxishi', 'has_stock',                 # 其他关键字 
    'has_plzread',                                          # 其他关键字
]

NUM_TOPKW = 1500

# init data, skf
df_train = pd.read_csv(join(PATH_DATA_ROOT, PATH_DATA_TRAIN), encoding='utf-8')
df_test = pd.read_csv(join(PATH_DATA_ROOT, PATH_DATA_TEST), encoding='utf-8')
skf = StratifiedKFold(n_splits=SKF_SPLIT, shuffle=True, random_state=RANDOM_STATE)


# preprocess (入模前的最终处理, 若需要加工特征等, 使用data文件夹下的preprocess.ipynb)
df_train['content_noengnum'] = df_train['content_noengnum'].astype(str)
df_test['content_noengnum'] = df_test['content_noengnum'].astype(str)
df_x_train, df_x_test, y, le = Preprocess.topk(df_train, df_test, 
                                               ADDITIONAL_FEATURES, 
                                               'content_noengnum',
                                               NUM_TOPKW, 
                                               allowPOS=('n', 'v', 'vn', 'ns', 'nr', 'nt', 'nz'))
# df_x_train, df_x_test, y, le = Preprocess.tfidf(df_train, df_test,
#                                                 ADDITIONAL_FEATURES,
#                                                 'content')


# train (xgb w/ skf)
res = []

# iterate over the folds
for i, (train_idx, val_idx) in tqdm(enumerate(skf.split(df_x_train, y))):
    train_x = df_x_train.loc[train_idx]
    train_y = y[train_idx]
    val_x = df_x_train.loc[val_idx]
    val_y = y[val_idx]
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dval = xgb.DMatrix(val_x, label=val_y)


    # define the XGBoost parameters
    def xgb_eval(params):
        # train the XGBoost model
        model = xgb.train(
            {
                'objective': 'multi:softmax',
                'num_class': len(np.unique(y)),
                'disable_default_eval_metric': True,
                'tree_method': 'gpu_hist',

                'learning_rate': params['learning_rate'],
                'max_depth': params['max_depth'],
                'min_child_weight': params['min_child_weight'],
                'gamma': params['gamma'],
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
            },
            dtrain,
            custom_metric=XgbCustomEval.f1_macro,
            evals=[(dval, 'val')],
            num_boost_round=2000,
            early_stopping_rounds=50,
        )

        # evaluate the model
        val_y_pred = model.predict(dval)

        score = -f1_score(val_y, val_y_pred, average='macro')

        return score
    
    # define the hyperparameter space
    spaces = {
        "learning_rate": hp.loguniform("learning_rate",np.log(0.001),np.log(0.2)),
        "max_depth": hp.choice("max_depth",range(3,31)),
        "min_child_weight": hp.choice("min_child_weight",range(0,11)),
        "gamma": hp.uniform("gamma",0.0,1.0),
        "subsample": hp.uniform("subsample",0.5,1.0),
        "colsample_bytree": hp.uniform("colsample_bytree",0.5,1.0),
        "reg_alpha": hp.uniform("reg_alpha", 0.0, 0.5),
        "reg_lambda": hp.uniform("reg_lambda", 0.0, 0.5),
    }

    trials = Trials()
    best = fmin(fn=xgb_eval, space=spaces, algo= tpe.suggest, max_evals=100, trials=trials)

    best_params = space_eval(spaces,best)
    res.append(best_params)



pickle.dump(res, open(f'./tuning/result.pkl', 'wb'))
