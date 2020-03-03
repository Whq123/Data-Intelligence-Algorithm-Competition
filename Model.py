import numpy as np
import pandas as pd
import datetime
import random

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
# from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)
# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000
import os
# Setup cross validation folds
kf = KFold(n_splits=2, random_state=2019, shuffle=True)

# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='binary:logistic',
                       num_leaves=6,
                       learning_rate=0.005,
                       n_estimators=500,
                       early_stopping_rounds=200,
                       max_bin=200,
                       bagging_fraction=0.8,
                       bagging_freq=4,
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(booster = 'gbtree' ,
                       learning_rate=0.005,
                       n_estimators=500,
                       early_stopping_rounds=200,
                       verbose_eval=100,
                       max_depth=5,
                       subsample=0.7,
                       colsample_bytree=0.8,
                       objective='binary:logistic',
                       eval_metric ='logloss',
                       nthread=8,
                       silent = True,
                       scale_pos_weight=2.5)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=500,
                                early_stopping_rounds=200,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=500,
                          early_stopping_rounds=200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
# stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
#                                 meta_regressor=xgboost,
#                                 use_features_in_secondary=True)

#--------------------------------------------------------------------------------

last = pd.read_csv(r'G:\三枪一炮\YunJifen\last_data2')
all = pd.read_csv(r'G:\三枪一炮\YunJifen\all_data2')

feature = ['customer_id', 'customer_counts', 'customer_province', 'customer_city',
           'long_time', 'is_customer_rate_num', 'isnot_customer_rate_num',
           'is_member_actived', 'is_customer_have_discount_count',
           'goods_price_max', 'goods_price_min', 'goods_price_mean',
           'goods_price_std', 'goods_has_discount_counts',
           'goods_hasnot_discount_counts', 'goods_status_1', 'goods_status_2',
           'order_amount_sum', 'order_total_payment_sum',
           'order_total_discount_mean', 'order_status_max',
           'order_detail_amount_sum', 'order_detail_payment_sum',
           'detail_discount_div_amount', 'order_detail_status_max',
           'order_pay_dayofyear_max', 'order_pay_dayofyear_min',
           'order_pay_rate_all', 'order_pay_rate_month', 'order_pay_rate_week',
           'order_pay_gap_max', 'order_pay_gap_min', 'order_amount_month_max',
           'order_amount_month_min', 'order_pay_day_max', 'order_pay_day_mean',
           'last_one_month_count', 'last_one_month_buyfreq',
           'last_ten_day_count', 'last_ten_day_buyfreq',
           'last_twoten_day_count', 'last_twoten_day_buyfreq',
           'last_two_month_count', 'last_three_month_count',
           'last_three_month_buyfreq', 'first_one_month_count',
           'first_two_month_count', 'first_three_month_count',
           'order_amount_month_sum', 'goods_id_kinds_count',
           'order_total_payment_month_sum', 'order_total_discount_month_sum',
           'order_pay_dayofyear_gap','label']

train_labels= last['label']
X_train = last[feature]

# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model,X=X_train):
    # rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error"))
    return (rmse)

scores = {}
# xgboost: 0.1207 (0.0003) -- 0.79
# xgboost: 0.0713 (0.0278)
score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())
xgb_model_full_data = xgboost.fit(X_train, train_labels)


score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())
lgb_model_full_data = lightgbm.fit(X_train, train_labels)

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())
svr_model_full_data = svr.fit(X_train, train_labels)

score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())
ridge_model_full_data = ridge.fit(X_train, train_labels)

score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())
rf_model_full_data = rf.fit(X_train, train_labels)

score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())
gbr_model_full_data = gbr.fit(X_train, train_labels)

# Fit the model
# stack_gen_model = stack_gen.fit(np.array(X_train), np.array(train_labels))
# Blend models in order to make the final predictions more robust to overfitting
# (0.1 * ridge_model_full_data.predict(X)) + \

def blended_predictions(X):
    return (
            (0.2 * svr_model_full_data.predict(X)) + \
            (0.1 * gbr_model_full_data.predict(X)) + \
            (0.1 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.05 * rf_model_full_data.predict(X)) + \

# blended_score = rmsle(train_labels, blended_predictions(X))

feature.pop(feature.index('label'))  # 删除 label
X_test = all[feature]
result = xgb_model_full_data.predict(X_test)
# 保存 xgb模型
# clf.save_model('./xgb.model')
# load model
# bst2 = xgb.Booster(model_file='xgb.model1')

result = blended_predictions(X_test)
res = all[['customer_id']]
res['result'] = result

train = pd.read_csv(r'G:\三枪一炮\YunJifen\second_contest\data2\round2_diac2019_train.csv',
                    parse_dates=['order_pay_time', 'goods_list_time', 'goods_delist_time'] )
train = pd.DataFrame(train)
train.loc[train['order_detail_amount'].isnull(), 'order_detail_amount'] = train['order_detail_discount'] + train['order_detail_payment']
data = pd.DataFrame(train[['customer_id']]).drop_duplicates(['customer_id']).dropna()
data = (data.merge(res,on=['customer_id'],how='left')).sort_values(['customer_id'])
data['customer_id'] = data['customer_id'].astype('int64')
data['result'] = data['result'].fillna(0)
result = data[['customer_id','result']]
result.to_csv('./round2_diac2019_test.csv', index=False)


