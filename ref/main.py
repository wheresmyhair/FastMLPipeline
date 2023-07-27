import numpy as np
import pandas as pd
import lightgbm as lgb

from config import param_dict
from tqdm import tqdm
from os.path import (join)
from utils import (Preprocess, CustomEval, report_model)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, f1_score, classification_report)


notes = ''


CURRENT_TIME = param_dict['current_time']
PATH_DATA_ROOT = param_dict['path_data_root']
PATH_OUT = param_dict['path_out']
PATH_DATA_TRAIN = param_dict['path_data_train']
PATH_DATA_TEST = param_dict['path_data_test']
PATH_CUSTOM_DICT = param_dict['path_custom_dict']
PATH_STOP_WORDS = param_dict['path_stop_words']
COLNAMES_NLP = param_dict['colnames_nlp']
SKF_SPLIT = param_dict['skf_split']
RANDOM_STATE = param_dict['random_state']
LGB_PARAMS = param_dict['lgb_params']


# init data, skf
df_train = pd.read_csv(join(PATH_DATA_ROOT, PATH_DATA_TRAIN), encoding='utf-8')
df_test = pd.read_csv(join(PATH_DATA_ROOT, PATH_DATA_TEST), encoding='utf-8')
skf = StratifiedKFold(n_splits=SKF_SPLIT, shuffle=True, random_state=RANDOM_STATE)
manual_features = [x for x in df_train.columns if x.startswith('manualfeat_')]
stopwords = [x.strip() for x in open(PATH_STOP_WORDS, 'r', encoding='utf-8').readlines()]

idx_delete = [6251, 351, 5709, 6213, 6202, 6169, 312]
df_train = df_train.drop(idx_delete).reset_index(drop=True)

idx_to_celue = [5566, 5691, 5813, 5889, 5643, 5611]
df_train.loc[idx_to_celue, 'label'] = 2

idx_to_hangye = [6164]
df_train.loc[idx_to_hangye, 'label'] = 3

idx_to_jingong = [6253]
df_train.loc[idx_to_jingong, 'label'] = 7

idx_to_hongguan = [6127]
df_train.loc[idx_to_hongguan, 'label'] = 1

idx_to_chenbao = [6294]
df_train.loc[idx_to_chenbao, 'label'] = 0




# preprocess (NLP before modelling. If wants to add manual features, 
# please refer to `preprocess.ipynb` in `data` folder.)
df_train[COLNAMES_NLP] = df_train[COLNAMES_NLP].astype(str)
df_test[COLNAMES_NLP] = df_test[COLNAMES_NLP].astype(str)
pp = Preprocess(df_train, df_test, 'label', manual_features, encode_y=True, custom_vocab=PATH_CUSTOM_DICT)
pp.tfidf(colnames_nlp=COLNAMES_NLP, mode='seperate', stop_words=None)
pp.ngram(colnames_nlp=COLNAMES_NLP, mode='seperate', stop_words=None)
df_x_train, df_x_test, y, le, colnames_encode = pp.finalize('all', encode_columns=True)
colnames_decode = {v: k for k, v in colnames_encode.items()}
print(f'[INFO] Train shape: {df_x_train.shape}')
colnames_decode['1']
print(f'[INFO] Test shape: {df_x_test.shape}')

# df_x_train['manualfeat_kw_策略点评'] = df_x_train['content'].apply(lambda x: 1 if '策略点评' in x else 0)
# df_x_test['manualfeat_kw_策略点评'] = df_x_test['content'].apply(lambda x: 1 if '策略点评' in x else 0)

# df_x_train['manualfeat_kw_证券策略'] = df_x_train['content'].apply(lambda x: 1 if '证券策略' in x else 0)
# df_x_test['manualfeat_kw_证券策略'] = df_x_test['content'].apply(lambda x: 1 if '证券策略' in x else 0)

# df_x_train['manualfeat_kw_策略聚焦'] = df_x_train['content'].apply(lambda x: 1 if '策略聚焦' in x else 0)
# df_x_test['manualfeat_kw_策略聚焦'] = df_x_test['content'].apply(lambda x: 1 if '策略聚焦' in x else 0)

# df_x_train['manualfeat_kw_中泰策略'] = df_x_train['content'].apply(lambda x: 1 if '中泰策略' in x else 0)
# df_x_test['manualfeat_kw_中泰策略'] = df_x_test['content'].apply(lambda x: 1 if '中泰策略' in x else 0)

# df_x_train['manualfeat_kw_专题策略'] = df_x_train['content'].apply(lambda x: 1 if '专题策略' in x else 0)
# df_x_test['manualfeat_kw_专题策略'] = df_x_test['content'].apply(lambda x: 1 if '专题策略' in x else 0)


# train (w/ skf)
res = []
# iterate over the folds
for i, (train_idx, val_idx) in tqdm(enumerate(skf.split(df_x_train, y))):
    train_x = df_x_train.loc[train_idx]
    train_y = y[train_idx]
    val_x = df_x_train.loc[val_idx]
    val_y = y[val_idx]

    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    evals_result = {}
    # train the XGBoost model
    model = lgb.train(
        LGB_PARAMS, 
        dtrain, 
        valid_sets=[dval, dtrain], 
        valid_names=['val', 'train'],
        feval=CustomEval.f1_macro_lgb,
        callbacks=[
            lgb.early_stopping(50),
            lgb.record_evaluation(evals_result),
        ],
        num_boost_round=2000,
    )

    # evaluate the model
    val_y_pred = model.predict(val_x, num_iteration=model.best_iteration)
    val_y_pred = np.argmax(val_y_pred, axis=1)

    score = f1_score(val_y, val_y_pred, average='macro')
    confmat = confusion_matrix(val_y, val_y_pred)
    report = classification_report(val_y, val_y_pred)
    print(confmat)
    print(score)
    print(report)
    
    # failed cases    
    df_failed = df_train.loc[val_idx[val_y != val_y_pred]]
    df_failed['pred'] = le.inverse_transform(val_y_pred[val_y != val_y_pred])
    df_failed['fold'] = i
    
    res.append({
        'split': i,
        'model': model,
        'f1_score': score,
        'confmat': confmat,
        'report': report,
        'evals_result': evals_result,
        'fails': df_failed,
    })


# report, feature importance and save model
additional_info = f'{notes}\nTrain shape: {str(df_x_train.shape)}\nTest shape: {str(df_x_test.shape)}\n'
report_model(PATH_OUT, res, colnames_decode, additional_info=additional_info)


# pred & correction
dfs_pred = pd.DataFrame()
probas = np.zeros((len(df_x_test), len(np.unique(y))))
probas_list = []
for model in tqdm(res):
    pred_proba = model['model'].predict(df_x_test, num_iteration=model['model'].best_iteration)
    probas += pred_proba
    probas_list.append(pred_proba)
    

probas = probas / len(res)
pred = le.inverse_transform(np.argmax(probas, axis=1))
df_test['label'] = pred


# save pred result
np.save(join(PATH_OUT, 'probas_list.npy'), probas_list)
df_test[['uid', 'label']].to_csv(join(PATH_OUT, 'submission.csv'),index=False, encoding='utf-8')
df_test.to_excel(join(PATH_OUT, 'vis_submission.xlsx'), index=False)