import numpy as np
import pandas as pd
import lightgbm as lgb

from config import param_dict
from tqdm import tqdm
from os.path import (join)
from utils import (Preprocess, report_model)
from sklearn.model_selection import StratifiedKFold


CURRENT_TIME = param_dict['current_time']
PATH_DATA_ROOT = param_dict['path_data_root']
PATH_OUT = param_dict['path_out']
PATH_DATA_TRAIN = param_dict['path_data_train']
PATH_DATA_TEST = param_dict['path_data_test']
PATH_CUSTOM_DICT = param_dict['path_custom_dict']
PATH_STOP_WORDS = param_dict['path_stop_words']
COLNAME_NLP = param_dict['colname_nlp']
SKF_SPLIT = param_dict['skf_split']
RANDOM_STATE = [2023,6666,8888,9999,1]
LGB_PARAMS = param_dict['lgb_params']


# init data, skf
df_train = pd.read_csv(join(PATH_DATA_ROOT, PATH_DATA_TRAIN), encoding='utf-8')
df_test = pd.read_csv(join(PATH_DATA_ROOT, PATH_DATA_TEST), encoding='utf-8')
skf = StratifiedKFold(n_splits=SKF_SPLIT, shuffle=True, random_state=RANDOM_STATE)
manual_features = [x for x in df_train.columns if x.startswith('manualfeat_')]
stopwords = [x.strip() for x in open(PATH_STOP_WORDS, 'r', encoding='utf-8').readlines()]


# preprocess (NLP before modelling. If wants to add manual features, 
# please refer to `preprocess.ipynb` in `data` folder.)
df_train[COLNAME_NLP] = df_train[COLNAME_NLP].astype(str)
df_test[COLNAME_NLP] = df_test[COLNAME_NLP].astype(str)
pp = Preprocess(df_train, df_test, 'label', manual_features, encode_y=True, custom_vocab=PATH_CUSTOM_DICT)
pp.tfidf(colname_nlp=COLNAME_NLP, mode='seperate', stop_words=None)
pp.ngram(colname_nlp=COLNAME_NLP, mode='seperate', stop_words=None)
df_x_train, df_x_test, y, le, colnames_encode = pp.finalize('all', encode_columns=True)
colnames_decode = {v: k for k, v in colnames_encode.items()}
print(f'[INFO] Train shape: {df_x_train.shape}')
print(f'[INFO] Test shape: {df_x_test.shape}')


dtrain = lgb.Dataset(df_x_train, label=y)
# train over seeds
res = []
for seed in RANDOM_STATE:
    LGB_PARAMS['seed'] = seed  # override seed
    # train
    model = lgb.train(
        LGB_PARAMS, 
        dtrain, 
        num_boost_round=90,
    )

    res.append({
        'seed': seed,
        'model': model,
    })


# feature importance and save model
report_model(PATH_OUT, res, colnames_decode, no_val_mode=True)


# pred & correction
dfs_pred = pd.DataFrame()
probas = np.zeros((len(df_x_test), len(np.unique(y))))
for model in tqdm(res):
    probas += model['model'].predict(df_x_test, num_iteration=model['model'].best_iteration)

probas = probas / len(res)
pred = le.inverse_transform(np.argmax(probas, axis=1))
df_test['label'] = pred
df_test.loc[(df_test['manualfeat_chenbao'] == 1), 'label'] = 0


# save pred result
df_test[['uid', 'label']].to_csv(join(PATH_OUT, 'submission.csv'),index=False, encoding='utf-8')
df_test.to_excel(join(PATH_OUT, 'vis_submission.xlsx'), index=False)