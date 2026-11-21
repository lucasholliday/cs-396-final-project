"""Small-data soft outcome-regime experiment; threshold is a carry proxy, not contact truth."""
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.linear_model import LogisticRegression,Ridge

class SoftCarryRegressor(RegressorMixin,BaseEstimator):
 def __init__(self,threshold=60.,high_regression=False):self.threshold=threshold;self.high_regression=high_regression
 def fit(self,X,y):
  X=np.asarray(X);y=np.asarray(y);low=y<self.threshold;self.global_mean_=float(np.mean(y));self.low_mean_=float(np.mean(y[low])) if low.any() else self.global_mean_;self.high_mean_=float(np.mean(y[~low])) if (~low).any() else self.global_mean_;self.classifier_=None;self.regressor_=None
  if len(np.unique(low))==2:
   self.classifier_=LogisticRegression(C=.3,max_iter=1000,random_state=396).fit(X,low.astype(int))
   if self.high_regression and (~low).sum()>=8:self.regressor_=Ridge(alpha=50).fit(X[~low],y[~low])
  return self
 def predict(self,X):
  X=np.asarray(X)
  if self.classifier_ is None:return np.full(len(X),self.global_mean_)
  p=self.classifier_.predict_proba(X)[:,list(self.classifier_.classes_).index(1)];high=np.maximum(self.threshold,self.regressor_.predict(X)) if self.regressor_ is not None else self.high_mean_
  return p*self.low_mean_+(1-p)*high

class QualityRoutedRegressor(RegressorMixin,BaseEstimator):
 """Use a launch model when its two inputs exist; otherwise a separately fitted fallback.
 Routing depends only on feature availability. All branch transforms fit on training data.
 """
 def __init__(self,primary='svr10',fallback='ridge10'):self.primary=primary;self.fallback=fallback
 @staticmethod
 def branch(name):
  from sklearn.pipeline import make_pipeline
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler
  from sklearn.dummy import DummyRegressor
  from sklearn.ensemble import ExtraTreesRegressor
  from sklearn.svm import SVR
  from sklearn.compose import TransformedTargetRegressor
  if name=='mean':e=DummyRegressor(strategy='mean')
  elif name.startswith('ridge'):e=Ridge(alpha=float(name[5:]))
  elif name.startswith('svr'):e=TransformedTargetRegressor(regressor=SVR(C=float(name[3:]),epsilon=.1),transformer=StandardScaler())
  else:e=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=4,random_state=396,n_jobs=1)
  return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),e)
 @staticmethod
 def fallback_x(X):return X[:,2:] if X.shape[1]>2 else np.ones((len(X),1))
 def fit(self,X,y):
  X=np.asarray(X,float);y=np.asarray(y);valid=np.isfinite(X[:,:2]).all(1);self.fallback_=self.branch(self.fallback).fit(self.fallback_x(X),y);self.primary_=None
  if valid.sum()>=8:self.primary_=self.branch(self.primary).fit(X[valid,:2],y[valid])
  return self
 def predict(self,X):
  X=np.asarray(X,float);p=np.asarray(self.fallback_.predict(self.fallback_x(X)));valid=np.isfinite(X[:,:2]).all(1)
  if self.primary_ is not None and valid.any():p[valid]=self.primary_.predict(X[valid,:2])
  return p
