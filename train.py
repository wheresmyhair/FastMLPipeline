import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class MyStacking:
    
    def __init__(self, estimators, final_estimator, cv=5, random_state=2023):
        self.cv = cv
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.random_state = random_state

        self.skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
    def fit(self, X, y):
        dataset_train = self.stacking(X, y)

        self.final_estimator.fit(dataset_train, y)

    def stacking(self, X, y):
        dataset_train = np.zeros((X.shape[0], len(self.estimators)))
        for i, model in enumerate(self.estimators):
            for (train, val) in kf.split(X, y):
                X_train = X[train]
                X_val = X[val]
                y_train = y[train]

                y_val_pred = model.fit(X_train, y_train).predict(X_val)
                dataset_train[val, i] = y_val_pred
            self.estimators[i] = model
        return dataset_train

    # 模型预测
    def predict(self, X):
        datasets_test = np.zeros((X.shape[0], len(self.estimators)))
        for i, model in enumerate(self.estimators):
            datasets_test[:, i] = model.predict(X)

        return self.final_estimator.predict(datasets_test)

    # 模型精度
    def score(self, X, y):
        datasets_test = np.zeros((X.shape[0], len(self.estimators)))
        for i, model in enumerate(self.estimators):
            datasets_test[:, i] = model.predict(X)
        return self.final_estimator.score(datasets_test, y)


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=0)

    estimators = [
        RandomForestClassifier(n_estimators=10),
        GradientBoostingClassifier(n_estimators=10)
    ]

    clf = MyStacking(estimators=estimators,
                     final_estimator=LogisticRegression())

    clf.fit(X_train, y_train)

    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))


##############################
# test module
import config as cfg
import catboost as cb
import xgboost as xgb
import lightgbm as lgb


root_path = '../_data/tianchi-loandefaulter'
df_raw = pd.read_csv(f'{root_path}/train.csv')
df_test = pd.read_csv(f'{root_path}/test.csv')
df_raw.drop(['id','issueDate','postCode','earliesCreditLine'], axis=1, inplace=True)
df_test.drop(['id','issueDate','postCode','earliesCreditLine'], axis=1, inplace=True)
col_cate = [
    'grade',
    'subGrade',
    'homeOwnership',
    'employmentTitle',
    'verificationStatus',
    'purpose',
    'regionCode',
    'initialListStatus',
    'applicationType',
    'title',
    'policyCode',
    'employmentLength'
]
for col in df_test.columns:
    df_raw[col].fillna(df_raw[col].mode()[0], inplace=True)
    df_test[col].fillna(df_test[col].mode()[0], inplace=True)
    if col in col_cate:
        df_raw[col] = df_raw[col].astype('str').astype('category')
        df_test[col] = df_test[col].astype('str').astype('category')
df_raw_x = df_raw.drop(['isDefault'], axis=1)
df_raw_y = df_raw['isDefault']


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)
clf_cb = cb.CatBoostClassifier(cat_features=col_cate, **cfg.CB_PARAMS)
clf_lgb = lgb.LGBMClassifier(**cfg.LGB_PARAMS)
clf_xgb = xgb.XGBClassifier(**cfg.XGB_PARAMS)
estimators = [clf_cb]


df_intermediate_train = np.zeros((df_raw.shape[0], len(estimators)))
df_intermediate_test = np.zeros((df_test.shape[0], len(estimators)))


for idx_skf, (idx_train, idx_val) in enumerate(skf.split(df_raw_x, df_raw_y)):
    df_train_x = df_raw_x.loc[idx_train]
    df_train_y = df_raw_y[idx_train]
    df_val_x = df_raw_x.loc[idx_val]
    df_val_y = df_raw_y[idx_val]

    for idx_model, model in enumerate(estimators):
        model_fitted = model.fit(X=df_train_x, y=df_train_y,
                                 eval_set=[(df_val_x, df_val_y)])
        
        df_intermediate_train[idx_val, idx_model] = model_fitted.predict_proba(df_val_x)[:,1]
        df_intermediate_test[:, idx_model] += model_fitted.predict_proba(df_test)[:,1] / skf.n_splits
    
    
    # model_cb = clf_cb.fit(X=df_train_x, y=df_train_y,
    #                       eval_set=(df_val_x, df_val_y))
    
    
    # classification_report(df_val_y, model_lgb.predict(df_val_x))
    


    
dataset_train = np.zeros((X.shape[0], len(self.estimators)))


for i, model in enumerate(self.estimators):
    for (train, val) in kf.split(X, y):
        X_train = X[train]
        X_val = X[val]
        y_train = y[train]

        y_val_pred = model.fit(X_train, y_train).predict(X_val)
        dataset_train[val, i] = y_val_pred
    self.estimators[i] = model


# 模型预测
def predict(self, X):
    datasets_test = np.zeros((X.shape[0], len(self.estimators)))
    for i, model in enumerate(self.estimators):
        datasets_test[:, i] = model.predict(X)

    return self.final_estimator.predict(datasets_test)

# 模型精度
def score(self, X, y):
    datasets_test = np.zeros((X.shape[0], len(self.estimators)))
    for i, model in enumerate(self.estimators):
        datasets_test[:, i] = model.predict(X)
    return self.final_estimator.score(datasets_test, y)
