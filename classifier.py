from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        self.n_components = 200
        self.n_estimators = 300
        clf1 = XGBClassifier(learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
        self.clf = Pipeline([
            ('pca', PCA(n_components=self.n_components)), 
                
            ('clf',clf1)
        ])

    def fit(self, X, y): 		
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
