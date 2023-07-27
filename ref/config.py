from datetime import datetime
from os import (makedirs)
from shutil import (copyfile)

PATH_DATA_ROOT = '../data_0512/csv/out/'
PATH_DATA_TRAIN = 'pp_train.csv'
PATH_DATA_TEST = 'pp_test.csv'
PATH_CUSTOM_DICT = './vocabs/vocab_custom.txt'
PATH_STOP_WORDS = '../data_0512/vocab/stopwords.txt'
COLNAMES_NLP = ['content']
SKF_SPLIT = 5
RANDOM_STATE = 2023
LGB_PARAMS = {
    'num_class': 10,
    'device_type': 'cpu',
    'num_threads': 24,
    'objective': 'softmax',
    'num_leaves': 127,
    'force_col_wise': True,
    'seed': RANDOM_STATE,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
}

CURRENT_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
PATH_OUT = f'./runs/{CURRENT_TIME}'
makedirs(PATH_OUT, exist_ok=True)

#######################################################

param_dict = {
    'current_time': CURRENT_TIME,
    'path_data_root': PATH_DATA_ROOT,
    'path_out': PATH_OUT,
    'path_data_train': PATH_DATA_TRAIN,
    'path_data_test': PATH_DATA_TEST,
    'path_custom_dict': PATH_CUSTOM_DICT,
    'path_stop_words': PATH_STOP_WORDS,
    'colnames_nlp': COLNAMES_NLP,
    'skf_split': SKF_SPLIT,
    'random_state': RANDOM_STATE,
    'lgb_params': LGB_PARAMS,
}

copyfile('./config.py', f'{PATH_OUT}/config.py')
copyfile('../data_0512/preprocess.ipynb', f'{PATH_OUT}/preprocess.ipynb')
print(f'[INFO] Config file copied to {PATH_OUT}/config.py')