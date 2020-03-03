import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

import datetime
import random
import seaborn as sns
import matplotlib. pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")


last_data = pd.read_csv(r'G:\三枪一炮\YunJifen\last_data2')
all_data = pd.read_csv(r'G:\三枪一炮\YunJifen\all_data2')
last = last_data.copy()
all = all_data.copy()

# 异常值填充 ，求众数,填充
nums = all['order_pay_day_mean']  # 11
mode = sorted(nums)[len(nums) // 2]
all.loc[all['order_pay_day_mean'].isnull(),'order_pay_day_mean'] = 11.0

nums = all['order_pay_day_max']  # 15.0
mode = sorted(nums)[len(nums) // 2]
all.loc[all['order_pay_day_max'].isnull(),'order_pay_day_max'] = 15.0


nums = all['order_amount_month_min']  # 1.0
mode = sorted(nums)[len(nums) // 2]
all.loc[all['order_amount_month_min'].isnull(),'order_amount_month_min'] = 1.0

nums = all['order_amount_month_max']  # 1.0
mode = sorted(nums)[len(nums) // 2]
all.loc[all['order_amount_month_max'].isnull(),'order_amount_month_max'] = 1.0

all.loc[all['detail_discount_div_amount']< 0, 'detail_discount_div_amount' ]= 0
all.loc[all['goods_status_2']< 0, 'goods_status_2' ] = 0  # 特殊异常值，商品状态编码为 众数代替
last.loc[last['detail_discount_div_amount']< 0, 'detail_discount_div_amount' ]= 0
last.loc[last['goods_status_2']< 0, 'goods_status_2' ] = 0  # 特殊异常值，商品状态编码为 众数代替

# customer_counts 截取30
last.loc[last['customer_counts'] > 30, 'customer_counts'] = 30
# isnot_customer_rate_num 截取 30
last.loc[last['isnot_customer_rate_num'] > 30, 'isnot_customer_rate_num'] = 30
# goods_has_discount_counts 截取15
last.loc[last['goods_has_discount_counts'] > 15, 'goods_has_discount_counts'] = 15
# goods_hasnot_discount_counts 截取30
last.loc[last['goods_hasnot_discount_counts'] > 30, 'goods_hasnot_discount_counts'] = 30
# goods_status_1 截取 20  （问题）
last.loc[last['goods_status_1'] > 20, 'goods_status_1'] = 20

# goods_status_2 截取 25
last.loc[last['goods_status_2'] > 25, 'goods_status_2'] = 25
# order_amount_sum 截取 50000
last.loc[last['order_amount_sum'] > 50000, 'order_amount_sum'] = 50000
# order_total_payment_sum 截取 25000
last.loc[last['order_total_payment_sum'] > 25000, 'order_total_payment_sum'] = 25000
# order_status_max 101 改成6  (问题)
last.loc[last['order_status_max'] == 101, 'order_status_max'] = 6
# order_detail_amount_sum 截取 15000
last.loc[last['order_detail_amount_sum'] > 15000, 'order_detail_amount_sum'] = 15000
# order_detail_payment_sum 截取 15000
last.loc[last['order_detail_payment_sum'] > 15000, 'order_detail_payment_sum'] = 15000
# detail_discount_div_amount 截取 0 （问题）
last.loc[last['detail_discount_div_amount'] > 0, 'detail_discount_div_amount'] = 0
# order_detail_status_max 101 改成6
last.loc[last['order_detail_status_max'] == 101, 'order_detail_status_max'] = 6
# order_pay_rate_all 截取5
last.loc[last['order_pay_rate_all'] > 5, 'order_pay_rate_all'] = 5
# order_pay_rate_month 截取 4
last.loc[last['order_pay_rate_month'] > 4, 'order_pay_rate_month'] = 4
# order_pay_rate_week 截取 1
last.loc[last['order_pay_rate_week'] > 1, 'order_pay_rate_week'] = 1
# order_amount_month_max 截取 20
last.loc[last['order_amount_month_max'] > 20, 'order_amount_month_max'] = 20
# order_amount_month_min 截取 20
last.loc[last['order_amount_month_min'] > 20, 'order_amount_month_min'] = 20
# last_one_month_count 截取10
last.loc[last['last_one_month_count'] > 10, 'last_one_month_count'] = 10
# last_two_month_count 截取 10
last.loc[last['last_two_month_count'] > 10, 'last_two_month_count'] = 10
# last_three_month_count 截取30
last.loc[last['last_three_month_count'] > 30, 'last_three_month_count'] = 30
# first_three_month_count 截取20
last.loc[last['first_three_month_count'] > 20, 'first_three_month_count'] = 20
# first_two_month_count 截取20
last.loc[last['first_two_month_count'] > 20, 'first_two_month_count'] = 20
# first_one_month_count 截取20
last.loc[last['first_one_month_count'] > 20, 'first_one_month_count'] = 20
# order_amount_month_sum 截取10000
last.loc[last['order_amount_month_sum'] > 10000, 'order_amount_month_sum'] = 10000
# goods_id_kinds_count 截取30
last.loc[last['goods_id_kinds_count'] > 30, 'goods_id_kinds_count'] = 30
# order_total_payment_month_sum 截取2500
last.loc[last['order_total_payment_month_sum'] > 2500, 'order_total_payment_month_sum'] = 2500
# order_total_discount_month_sum 截取300
last.loc[last['order_total_discount_month_sum'] > 300, 'order_total_discount_month_sum'] = 300


# ---------------------------------------------------------------------------------------------------
# customer_counts 截取30
all.loc[all['customer_counts'] > 30, 'customer_counts'] = 30
# isnot_customer_rate_num 截取 30
all.loc[all['isnot_customer_rate_num'] > 30, 'isnot_customer_rate_num'] = 30
# goods_has_discount_counts 截取15
all.loc[all['goods_has_discount_counts'] > 15, 'goods_has_discount_counts'] = 15
# goods_has_not_discount_counts 截取30
all.loc[all['goods_hasnot_discount_counts'] > 30, 'goods_hasnot_discount_counts'] = 30
# goods_status_1 截取 20  （问题）
all.loc[all['goods_status_1'] > 20, 'goods_status_1'] = 20
# goods_status_2 截取 25
all.loc[all['goods_status_2'] > 25, 'goods_status_2'] = 25
# order_amount_sum 截取 50000
all.loc[all['order_amount_sum'] > 50000, 'order_amount_sum'] = 50000
# order_total_payment_sum 截取 25000
all.loc[all['order_total_payment_sum'] > 25000, 'order_total_payment_sum'] = 25000
# order_status_max 101 改成6  (问题)
all.loc[all['order_status_max'] == 101, 'order_status_max'] = 6
# order_detail_amount_sum 截取 15000
all.loc[all['order_detail_amount_sum'] > 15000, 'order_detail_amount_sum'] = 15000
# order_detail_payment_sum 截取 15000
all.loc[all['order_detail_payment_sum'] > 15000, 'order_detail_payment_sum'] = 15000
# detail_discount_div_amount 截取 0 （问题）
all.loc[all['detail_discount_div_amount'] > 0, 'detail_discount_div_amount'] = 0
# order_detail_status_max 101 改成6
all.loc[all['order_detail_status_max'] == 101, 'order_detail_status_max'] = 6
# order_pay_rate_all 截取5
all.loc[all['order_pay_rate_all'] > 5, 'order_pay_rate_all'] = 5
# order_pay_rate_month 截取 4
all.loc[all['order_pay_rate_month'] > 4, 'order_pay_rate_month'] = 4
# order_pay_rate_week 截取 1
all.loc[all['order_pay_rate_week'] > 1, 'order_pay_rate_week'] = 1
# order_amount_month_max 截取 20
all.loc[all['order_amount_month_max'] > 20, 'order_amount_month_max'] = 20
# order_amount_month_min 截取 20
all.loc[all['order_amount_month_min'] > 20, 'order_amount_month_min'] = 20
# all_one_month_count 截取10
all.loc[all['last_one_month_count'] > 10, 'last_one_month_count'] = 10
# all_two_month_count 截取 10
all.loc[all['last_two_month_count'] > 10, 'last_two_month_count'] = 10
# all_three_month_count 截取30
all.loc[all['last_three_month_count'] > 30, 'last_three_month_count'] = 30
# first_three_month_count 截取20
all.loc[all['first_three_month_count'] > 20, 'first_three_month_count'] = 20
# first_two_month_count 截取20
all.loc[all['first_two_month_count'] > 20, 'first_two_month_count'] = 20
# first_one_month_count 截取20
all.loc[all['first_one_month_count'] > 20, 'first_one_month_count'] = 20
# order_amount_month_sum 截取10000
all.loc[all['order_amount_month_sum'] > 10000, 'order_amount_month_sum'] = 10000
# goods_id_kinds_count 截取30
all.loc[all['goods_id_kinds_count'] > 30, 'goods_id_kinds_count'] = 30
# order_total_payment_month_sum 截取2500
all.loc[all['order_total_payment_month_sum'] > 2500, 'order_total_payment_month_sum'] = 2500
# order_total_discount_month_sum 截取300
all.loc[all['order_total_discount_month_sum'] > 300, 'order_total_discount_month_sum'] = 300

last.to_csv('./last_data2_deal')
all.to_csv('./all_data2_deal')


# 离差标准化

# ---------------------------------------------------------

"""
绘制各个属性的三点图，观察数据的基本特征

"""
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(last['customer_id'], last['goods_price_mean'])
plt.show()

# 添加修改 特征，
#  < 0.8 counts , > 0.8 counts
last['customer_counts_ll'] = 1
last.loc[last['customer_counts'] < 0.8, 'customer_counts_ll']= 0
last.loc[last['is_member_actived'] > 0, 'is_member_actived']= 1

all['customer_counts_ll'] = 1
all.loc[all['customer_counts'] < 0.8, 'customer_counts_ll']= 0
all.loc[all['is_member_actived'] > 0, 'is_member_actived']= 1




feature = ['customer_id', 'customer_counts', 'customer_counts_ll', 'customer_province', 'customer_city',
           'long_time', 'is_customer_rate_num', 'isnot_customer_rate_num',
           'is_member_actived', 'is_customer_have_discount_count',
           'goods_price_max', 'goods_price_min', 'goods_price_mean',
           'goods_price_std', 'goods_has_discount_counts',
           'goods_hasnot_discount_counts','goods_status_1','goods_status_2'
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
           'order_pay_dayofyear_gap','label' ]  #
# 解决倾斜特征
numeric_dtypes = ['int16', 'int32','int64','float16','float32','float64']
numeric = []
for i in last.columns:
    if last[i].dtype in numeric_dtypes:
        numeric.append(i)

# 为所有特征绘制箱型线图
sns.set_style('white')
f,ax = plt.subplots(figsize=(8,7))
ax.set_xscale('log')
ax = sns.boxplot(data= last[numeric], orient='h',palette='Set1')
ax.xaxis.grid(False)
ax.set(ylabel='Feature names')
ax.set(xlabel='Numeric values')
ax.set(title='Numeric Distribution of Features')
sns.despine(trim=True, left=True)

# 寻找偏弱的特征
from scipy.stats import skew, norm
skew_features =last[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
skewness = pd.DataFrame({'Skew' : high_skew})
skew_features.head(10)

# 用scipy函数boxcox1p来计算Box-Cox转换。我们的目标是找到一个简单的转换方式使数据规范化。
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
for i in skew_index:
    last[i] = boxcox1p(last[i], boxcox_normmax(last[i]+1))

# 处理所有的 skewed values
sns.set_style('white')
f, ax = plt.subplots(figsize=(8,7))
ax.set_xscale('log')
ax = sns.boxplot(data=last[skew_index], orient='h',palette='Set1')
ax.xaxis.grid(False)
ax.set(ylabel='Feature names')
ax.set(xlabel='Numeric values')
# --------------------------------------------------------------------------------------------------

numeric_dtypes = ['int16', 'int32','int64','float16','float32','float64']
numeric = []
for i in all.columns:
    if all[i].dtype in numeric_dtypes:
        numeric.append(i)

# 寻找偏弱的特征
from scipy.stats import skew, norm
skew_features =all[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
skewness = pd.DataFrame({'Skew' : high_skew})
skew_features.head(10)

# 用scipy函数boxcox1p来计算Box-Cox转换。我们的目标是找到一个简单的转换方式使数据规范化。
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
for i in skew_index:
    all[i] = boxcox1p(all[i], boxcox_normmax(all[i]+1))


last.to_csv('./last_data2_deal')
all.to_csv('./all_data2_deal')

# # 下采样 数据
def lower_sample_data(last_data, percent=1):
    '''
    percent:多数类别下采样的数量相对于少数类别样本数量的比例
    '''
    data0 = last_data[last_data['label'] == 0]  # 将多数类别的样本放在data0
    data1 = last_data[last_data['label'] == 1]  # 将少数类别的样本放在data1
    index = np.random.randint(
        len(data0), size=percent * (len(last_data) - len(data0)))  # 随机给定下采样取出样本的序号
    lower_data1 = data0.iloc[list(index)]  # 下采样
    return (pd.concat([lower_data1, data1]))
# ----------------------------------------------------------

last = pd.read_csv(r'G:\三枪一炮\YunJifen\last_data2_deal')
all = pd.read_csv(r'G:\三枪一炮\YunJifen\all_data2_deal')

# 构造 logs 特征， squares 特征
def logs(res,ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol = pd.Series(np.log(1.01+res[l])).values)
        res.columns.values[m] = l + '_log'
        m +=1
    return res
log_features = [ 'customer_counts','goods_price_mean',
   'order_amount_sum', 'order_total_payment_sum',
   'order_detail_amount_sum', 'order_detail_payment_sum',
   'order_pay_day_mean','order_amount_month_sum',
   'goods_id_kinds_count', 'order_total_payment_month_sum']

last_features = logs(last,log_features)
all_features = logs(all,log_features)

def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)
        res.columns.values[m] = l + '_sq'
        m +=1
    return res
squared_features =[ 'customer_counts','goods_price_mean',
   'order_total_discount_mean','order_pay_day_mean',
   'last_one_month_count', 'last_one_month_buyfreq', 'last_ten_day_count',
   'last_ten_day_buyfreq', 'last_twoten_day_count',
   'last_twoten_day_buyfreq', 'last_two_month_count',
   'last_three_month_count', 'last_three_month_buyfreq']
last_features = squares(last_features,squared_features)
all_features = squares(all_features,squared_features)

last_features.to_csv('./last_data2_add')
all_features.to_csv('./all_data2_add')