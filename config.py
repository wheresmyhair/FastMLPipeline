import os


RANDOM_STATE = 2023
LOG_PATH = f'./logs/'
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

LGB_PARAMS = {
    'seed': RANDOM_STATE,
    'device_type': 'cpu',
    'n_jobs':  -1,
    
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'force_col_wise': True,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'n_estimators': 2000,
    'early_stopping_round': 100,
}

XGB_PARAMS = {
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'enable_categorical': True,
    'tree_method': 'hist',
    
    'objective': 'binary:logistic',
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0.1, # L1 regularization
    'gamma': 30, # L2 regularization
}

CB_PARAMS = {
    'random_state': RANDOM_STATE,
    'thread_count': -1,
    'verbose': 10,
    
    # CatBoostClassifier will automatically detect the number of classes in multiclass classification
    'iterations': 2000,
    'early_stopping_rounds': 50,
    'learning_rate': 0.1,
    'depth': 7,
    'loss_function': 'Logloss',
    'l2_leaf_reg': 30,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
}

RF_PARAMS = {
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': True,
    
    'n_estimators': 2000,
    'max_depth': 7,
    'max_samples': 0.8,
    'max_features': 0.8,
}

DT_PARAMS = {
    'random_state': RANDOM_STATE,
    
    'max_depth': 7,
}

STACKING_PARAMS = {
    'cv': 5,
    'verbose': 2,
    'n_jobs': -1,
}