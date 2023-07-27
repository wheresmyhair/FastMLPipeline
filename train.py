import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


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
import lightgbm as lgb

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

clf_cb = cb.CatBoostClassifier(**cfg.CB_PARAMS)
clf_cb.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100, early_stopping_rounds=100)
clf_lgb = lgb.LGBMClassifier(**cfg.LGB_PARAMS)

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
