import logging
from datetime import datetime

import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import (StackingClassifier as STC,
                              RandomForestClassifier as RFC)
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.datasets import load_digits

import config as cfg


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger_file = logging.FileHandler(f"{cfg.LOG_PATH}{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger_file.setFormatter(logger_formatter)
logger = logging.getLogger()
logger.addHandler(logger_file)


# Load data
logger.info('Loading data...')
data = load_digits()
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=cfg.RANDOM_STATE)


# Initialize models
## Base models
logger.info('Initializing models...')
clf_xgb = xgb.XGBClassifier(**cfg.XGB_PARAMS)
clf_cb = cb.CatBoostClassifier(**cfg.CB_PARAMS)
clf_lgb = lgb.LGBMClassifier(**cfg.LGB_PARAMS)
clf_gnb = GNB()
clf_RFC = RFC(**cfg.RF_PARAMS)
clf_dtc = DTC(**cfg.DT_PARAMS)

estimators = [
    ('xgb', clf_xgb),
    ('cb', clf_cb),
    ('lgb', clf_lgb),
    ('gnb', clf_gnb),
    ('rfc', clf_RFC),
    ('dtc', clf_dtc)
]

## Meta model
final_estimator = RFC(**cfg.RF_PARAMS)

## Stacking model
clf = STC(estimators=estimators, final_estimator=final_estimator, **cfg.STACKING_PARAMS)


# Train
clf.fit(x_train, y_train)

clf.score(x_test, y_test)

clf.predict_proba(x_test)