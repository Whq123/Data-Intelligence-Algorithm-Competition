import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

def Computer_features(train_last, train_all, last_data, all_data):
    """一轮：     八个
        购买次数:                                customer_counts
        省份:                                    customer_province
        城市:                                    customer_city
        train 训练集中的 last - first 的时长:     long_time
        评价订单数量，用户统数统计（求和）:        is_customer_rate_num, isnot_customer_rate_num
        会员是否激活                              is_member_actived
        ---- 会员激活 & 商品可以打折扣 的订单求和：is_customer_have_discount_count
        性别        0 未知， 1男 ，2女             customer_gender_sum
    """
    for idx, data in enumerate([train_last, train_all]):
        customer_all = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()
        data = data.sort_values(by=['customer_id', 'order_pay_time'])
        data['count'] = 1
        # 一、购买次数
        tmp = data.groupby(['customer_id'])['count'].agg({'customer_counts': 'count'}).reset_index()
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        # 二、 省份 , last() 由迭代式获取其中的值， reset_index() 重置索引
        tmp = data.groupby(['customer_id'])['customer_province'].last().reset_index()
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        # 三、城市
        tmp = data.groupby(['customer_id'])['customer_city'].last().reset_index()
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        # 四、long_time ： 在train 训练集中的 last - first 的时长, order_pay_date_last : 统计这个用户的订单最后一次购买时间
        last_time = data.groupby(['customer_id'], as_index=False)['order_pay_time'].agg(
            {'order_pay_date_last': 'max', 'order_pay_date_first': 'min'}).reset_index()
        tmp['long_time'] = pd.to_datetime(last_time['order_pay_date_last']) - pd.to_datetime(last_time['order_pay_date_first'])
        tmp['long_time'] = tmp['long_time'].dt.days + 1
        del tmp['customer_city']
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 五，六 未评价订单数量，用户数统计（求和）
        data.loc[data['is_customer_rate'].isnull(), 'is_customer_rate'] = 0
        data['count'] = 1
        tmp1 = data.groupby(['customer_id'])['is_customer_rate'].agg({'is_customer_rate_num': 'sum'}).reset_index()
        customer_all = customer_all.merge(tmp1, on=['customer_id'], how='left')
        # 评价订单数量，用户数统计和
        c_count = data.groupby(['customer_id'])['count'].agg({'count': 'sum'}).reset_index()
        tmp2 = tmp1
        tmp2['is_customer_rate_num'] = pd.DataFrame(c_count['count'] - tmp1['is_customer_rate_num'])
        tmp2 = tmp2.rename(columns={'is_customer_rate_num': 'isnot_customer_rate_num'}, inplace=False)
        customer_all = customer_all.merge(tmp2, on=['customer_id'], how='left')
        data['count'] = 1

        # 七、会员是否激活  NaN -- 0, 按照customer_id 分组后，sum() ————> 1 == 0
        data.loc[data['is_member_actived'].isnull(), 'is_member_actived'] = 0
        tmp = data.groupby(['customer_id'], as_index=False)['is_member_actived'].agg(
            {'is_member_actived': 'sum'}).reset_index()
        tmp.loc[tmp['is_member_actived'] > 1, 'is_member_actived'] = 1
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 八 、 会员激活 & 商品可以打折扣 的订单求和
        tmp = data[(data['is_member_actived'] == 1) & (data['goods_has_discount'] == 1)].groupby(['customer_id'])[
            'count'].agg({'is_customer_have_discount_count': 'count'}).reset_index()
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        customer_all.loc[
            customer_all['is_customer_have_discount_count'].isnull(), 'is_customer_have_discount_count'] = 0

        # 九 顾客性别 customer_gender_sum
        data['count'] = 1
        data2 = data
        data.loc[data['customer_gender'].isnull(), 'customer_gender'] = 0
        data.loc[data['customer_gender'] == 2, 'customer_gender'] = -1
        tmp = data.groupby(['customer_id'], as_index=False)['customer_gender'].agg(
            {'customer_gender_sum': 'sum'}).reset_index()
        tmp.loc[tmp['customer_gender_sum'] > 1, 'customer_gender_sum'] = 1
        tmp.loc[tmp['customer_gender_sum'] < 0, 'customer_gender_sum'] = 2
        # 异常值用 3 代替
        tmp.loc[(data2['customer_gender'].isnull() & tmp['customer_gender_sum']==0),'customer_gender_sum'] = 3
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')


        if idx == 0:
            last_data = last_data.merge(customer_all, on='customer_id', how='left')
        else:
            all_data = all_data.merge(customer_all, on='customer_id', how='left')

    """
       二轮：  八个指标 
       3 三种 价格:goods_price_max, goods_price_min, goods_price_mean
       1 价格的方差: goods_price_std

       2 是否---折扣商品订单数量统计:  goods_has_discount_counts,  goods_has_not_discount_counts
       --- 2 商品状态 ： goods_status_1， goods_status_2
    """

    for idx, data in enumerate([train_last, train_all]):
        customer_all = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()
        # 一、二、三、价格
        tmp = data.groupby(['customer_id'], as_index=False)['goods_price'].agg(
            {'goods_price_max': 'max', 'goods_price_min': 'min', 'goods_price_mean': 'mean'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 四、goods_price 价格的方差
        tmp = data.groupby(['customer_id'], as_index=False)['goods_price'].agg({'goods_price_std': 'std'})
        # 只有一个订单的客户，商品价格的方差值 = 0
        tmp.loc[tmp['goods_price_std'].isnull(), 'goods_price_std'] = 0
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        del tmp['goods_price_std']
        # 五、六、未折扣商品订单数量统计  、 折扣商品订单数量统计
        tmp1 = data.groupby(['customer_id'])['goods_has_discount'].agg(
            {'goods_has_discount_counts': 'sum'})
        customer_all = customer_all.merge(tmp1, on=['customer_id'], how='left')
        data['count'] = 1
        c_count = data.groupby(['customer_id'])['count'].agg({'count': 'count'})
        tmp2 = tmp1
        tmp2['goods_has_discount_counts'] = pd.DataFrame(c_count['count'] - tmp1['goods_has_discount_counts'])
        tmp2 = tmp2.rename(columns={'goods_has_discount_counts': 'goods_hasnot_discount_counts'}, inplace=False)
        customer_all = customer_all.merge(tmp2, on=['customer_id'], how='left')
        # 七、八  商品状态 1, 2
        data.drop(data[(data['goods_status'] == 0)].index, inplace=True)
        data['count'] = 1
        data['goods_status'] = data['goods_status'] - 1
        tmp1 = data.groupby(['customer_id'])['goods_status'].agg({'goods_status_2': 'sum'})
        customer_all = customer_all.merge(tmp1, on=['customer_id'], how='left')
        tmp2 = tmp1.copy()
        tmp3 = data.groupby(['customer_id'])['goods_status'].agg({'goods_status_count': 'count'})
        tmp2['goods_status_2'] = pd.DataFrame(tmp3['goods_status_count'] - tmp1['goods_status_2'])
        tmp2 = tmp2.rename(columns={'goods_status_2': 'goods_status_1'}, inplace=False)
        customer_all = customer_all.merge(tmp2, on=['customer_id'], how='left')

        if idx == 0:
            last_data = last_data.merge(customer_all, on='customer_id', how='left')
        else:
            all_data = all_data.merge(customer_all, on='customer_id', how='left')

    """ 
      三轮：  八个
    父订单商品金额（求和）         ：order_amount_sum
    父订单实付金额（求和）         ：order_total_payment_sum
    父订单优惠金额/商品总金额       : order_total_discount_mean
    父订单的状态（最大值）        ：order_status_max
    父订单优惠金额（求和）：              order_detail_amount_sum
    父订单支付金额（求和）:               order_detail_payment_sum
    子订单优惠总金额/ 应付总金额 :     detail_discount_div_amount
    订单状态的 最大值:                order_detail_status_max
    """

    for idx, data in enumerate([train_last, train_all]):
        customer_all = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()
        # 一、父订单商品金额 （求和）
        tmp = data.groupby(['customer_id'])['order_amount'].agg({'order_amount_sum': 'sum'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        # 二 父订单实付金额（求和）
        tmp = data.groupby(['customer_id'])['order_total_payment'].agg(
            {'order_total_payment_sum': 'sum'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        # 三、父订单优惠金额 （求和）
        tmp = data.groupby(['customer_id'])['order_total_discount'].agg({'order_total_discount_mean': 'sum'})
        tmp2 = data.groupby(['customer_id'])['order_amount'].agg({'order_amount_sum': 'sum'})
        tmp['order_total_discount_mean'] = tmp['order_total_discount_mean'] / tmp2['order_amount_sum']
        tmp.loc[tmp['order_total_discount_mean'].isnull(), 'order_total_discount_mean'] = 0
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 四、父订单的状态 （最大值）
        tmp = data.groupby(['customer_id'])['order_status'].agg({'order_status_max': 'max'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 五、父订单优惠金额求和
        tmp = data.groupby(['customer_id'])['order_detail_amount'].agg({'order_detail_amount_sum': 'sum'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 六： 父订单支付金额求和
        tmp = data.groupby(['customer_id'])['order_detail_payment'].agg(
            {'order_detail_payment_sum': 'sum'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 七、子订单优惠 总金额求和
        # detail_discount_div_amount : order_detail_discount  子订单优惠金额 / order_detail_amount
        tmp = data.groupby(['customer_id'])['order_detail_discount'].agg({'detail_discount_div_amount': 'sum'})
        data.loc[data['order_detail_amount'].isnull(), 'order_detail_amount'] = data['order_detail_discount'] + data[
            'order_detail_payment']
        tmp2 = data.groupby(['customer_id'])['order_detail_amount'].agg({'order_detail_amount_sum': 'sum'})
        tmp['detail_discount_div_amount'] = tmp['detail_discount_div_amount'] / tmp2['order_detail_amount_sum']
        tmp.loc[tmp['detail_discount_div_amount'].isnull(), 'detail_discount_div_amount'] = 0
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 八、 订单状态的 最大值
        tmp = data.groupby(['customer_id'])['order_detail_status'].agg({'order_detail_status_max':'max'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        if idx == 0:
            last_data = last_data.merge(customer_all, on='customer_id', how='left')
        else:
            all_data = all_data.merge(customer_all, on='customer_id', how='left')

    """
        四轮：   11 个 
        一年中最后一天购买           order_pay_dayofyear_max
        一年中的第一天购买           order_pay_dayofyear_min
        整个周期内：购买次数/时间间隔 order_pay_rate_all
        月内：购买次数/时间间隔（平均）order_pay_rate_month
        周内：购买次数/时间间隔（平均）order_pay_rate_week
        最短购买的时间间隔           order_pay_gap_max
        最长购买的时间间隔           order_pay_gap_min
        每个月内的最后一次购买的时间（第几天）order_pay_day_max
        每个月内的最后一次购买的时间（平均） order_pay_day_mean
        每个月的购买次数最多  order_amount_month_max 
        每个月的购买次数最少  order_amount_month_min
    """

    for idx, data in enumerate([train_last, train_all]):
        customer_all = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()
        data = data.sort_values(by=['customer_id', 'order_pay_time'])
        data['count'] = 1

        tmp = data.groupby(['customer_id'])['count'].agg({'customer_counts': 'count'}).reset_index()
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 一年中最后一天购买order_pay_dayofyear_max
        # 一年中的第一天购买order_pay_dayofyear_min
        data['order_pay_dayofyear'] = data['order_pay_time'].dt.dayofyear
        tmp = data.groupby(['customer_id'])['order_pay_dayofyear'].agg(
            {'order_pay_dayofyear_max': 'max', 'order_pay_dayofyear_min': 'min'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 整个周期内：购买次数/时间间隔order_pay_rate_all
        tmp['order_pay_rate_all'] = customer_all['customer_counts'] / (
                tmp['order_pay_dayofyear_max'] - tmp['order_pay_dayofyear_min'])
        # tmp.replace([np.inf,-np.inf],np.nan)
        tmp.loc[(tmp['order_pay_dayofyear_max'] - tmp['order_pay_dayofyear_min']) == 0, 'order_pay_rate_all'] = 0
        tmp.loc[tmp['order_pay_rate_all'].isnull(), 'order_pay_rate_all'] = 0
        # tmp.loc[tmp['order_pay_rate_all'].isinf(),'order_pay_rate_all']=0
        del tmp['order_pay_dayofyear_max'], tmp['order_pay_dayofyear_min']
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        if idx == 0:
            month = 11
            week = 44
        else:
            month = 12
            week = 48

        # 月内：购买次数/时间间隔order_pay_rate_month
        tmp['order_pay_rate_month'] = customer_all['customer_counts'] / month
        tmp.loc[tmp['order_pay_rate_month'].isnull(), 'order_pay_rate_month'] = 0
        del tmp['order_pay_rate_all']
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 周内：购买次数/时间间隔order_pay_rate_week
        tmp['order_pay_rate_week'] = customer_all['customer_counts'] / week
        tmp.loc[tmp['order_pay_rate_week'].isnull(), 'order_pay_rate_week'] = 0
        del tmp['order_pay_rate_month']
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 最短购买的时间间隔order_pay_gap_max,order_pay_gap_min
        tmp = data.groupby(['customer_id'])['order_pay_gap'].agg(
            {'order_pay_gap_max': 'max', 'order_pay_gap_min': 'min'})
        tmp.loc[tmp['order_pay_gap_min'].isnull(), 'order_pay_gap_min'] = 0
        tmp.loc[tmp['order_pay_gap_max'].isnull(), 'order_pay_gap_max'] = 0
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 每个月的购买次数(最多，最少)(order_amount_month_max,order_amount_month_min)
        tmp1 = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()
        tmp2 = data.groupby(by=['customer_id', 'order_pay_month'])['count'].agg({'order_amount_month': 'count'})
        tmp3 = data.groupby(by=['customer_id', 'order_pay_month'])['order_pay_day'].agg(
            {'order_pay_day_big': 'max', 'order_pay_day_mid': 'mean'})
        tmp1 = tmp1.merge(tmp2, on=['customer_id'], how='left')
        tmp1 = tmp1.merge(tmp3, on=['customer_id'], how='left')

        tmp = tmp1.groupby(['customer_id'])['order_amount_month'].agg(
            {'order_amount_month_max': 'max', 'order_amount_month_min': 'min'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 每个月内的最后一次购买的时间（第几天）（平均）order_pay_day_max(order_pay_day_mean)
        del tmp['order_amount_month_max'], tmp['order_amount_month_min']
        tmp = tmp1.groupby(['customer_id'])['order_pay_day_big'].agg({'order_pay_day_max': 'max'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        del tmp['order_pay_day_max']
        tmp = tmp1.groupby(['customer_id'])['order_pay_day_mid'].agg({'order_pay_day_mean': 'mean'})
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')
        del customer_all['customer_counts']
        if idx == 0:
            last_data = last_data.merge(customer_all, on='customer_id', how='left')
        else:
            all_data = all_data.merge(customer_all, on='customer_id', how='left')

    """
----------------------------------------------------------------------------    
        五轮： 8个     
        最后一个月的购买次数  last_one_month_count
        
        最后 10 天的购买次数  last_ten_day_count
        最后 20 天的购买次数  last_twoten_day_count
        最后 10 天的购买频率（ 次数/10）last_ten_day_buyfreq
        最后 20 天的购买频率（ 次数/20）last_twoten_day_buyfreq
        
        最后一个月的购买频率（ 次数/30）last_one_month_buyfreq

        最后二个月的购买次数  last_two_month_count

        最后三个月的购买次数  last_three_month_count
        最后三个月的购买频率（次数/90） last_three_month_buyfreq

        第一个月的购买次数   first_one_month_count
        前二个月的购买次数   first_two_month_count
        前三个月的购买次数   first_three_month_count

    """
    for idx, data in enumerate([train_last, train_all]):
        customer_all = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()

        if idx == 0:
            # 最后一个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-01')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-09-02')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_one_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_one_month_count'].isnull(), 'last_one_month_count'] = 0
            # 最后一个月的购买频率（ 次数 / 30 ）
            tmp3['last_one_month_buyfreq'] = tmp3['last_one_month_count'] / 30
            del tmp3['last_one_month_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_one_month_buyfreq'].isnull(), 'last_one_month_buyfreq'] = 0

            # 最后 10 天的购买次数  last_ten_day_count
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-01')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-09-22')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_ten_day_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_ten_day_count'].isnull(), 'last_ten_day_count'] = 0
            # 最后 10 的购买频率（ 次数 / 10 ）
            tmp3['last_ten_day_buyfreq'] = tmp3['last_ten_day_count'] / 10
            del tmp3['last_ten_day_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_ten_day_buyfreq'].isnull(), 'last_ten_day_buyfreq'] = 0

            # 最后 20 天的购买次数  last_twoten_day_count
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-01')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-09-12')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_twoten_day_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_twoten_day_count'].isnull(), 'last_twoten_day_count'] = 0
            # 最后 20 的购买频率（ 次数 / 20 ）
            tmp3['last_twoten_day_buyfreq'] = tmp3['last_twoten_day_count'] / 20
            del tmp3['last_twoten_day_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_twoten_day_buyfreq'].isnull(), 'last_twoten_day_buyfreq'] = 0

            # 最后二个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-01')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-08-02')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_two_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_two_month_count'].isnull(), 'last_two_month_count'] = 0

            # 最后三个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-01')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-07-03')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_three_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_three_month_count'].isnull(), 'last_three_month_count'] = 0

            # 最后三个月的购买频率（次数 / 90）
            tmp3['last_three_month_buyfreq'] = tmp3['last_three_month_count'] / 90
            del tmp3['last_three_month_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_three_month_buyfreq'].isnull(), 'last_three_month_buyfreq'] = 0

            # 第一个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2012-12-01')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2012-11-01')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'first_one_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['first_one_month_count'].isnull(), 'first_one_month_count'] = 0

            # 前二个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2012-12-31')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2012-11-01')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'first_two_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['first_two_month_count'].isnull(), 'first_two_month_count'] = 0

            # 前三个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-01-29')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2012-11-01')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'first_three_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['first_three_month_count'].isnull(), 'first_three_month_count'] = 0

        else:
            # 最后一个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-31')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-10-01')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_one_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_one_month_count'].isnull(), 'last_one_month_count'] = 0
            # 最后一个月的购买频率（ 次数 / 30 ）
            tmp3['last_one_month_buyfreq'] = tmp3['last_one_month_count'] / 30
            del tmp3['last_one_month_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_one_month_buyfreq'].isnull(), 'last_one_month_buyfreq'] = 0

            # 最后 10 天的购买次数  last_ten_day_count
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-31')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-10-21')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_ten_day_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_ten_day_count'].isnull(), 'last_ten_day_count'] = 0
            # 最后 10 的购买频率（ 次数 / 10 ）
            tmp3['last_ten_day_buyfreq'] = tmp3['last_ten_day_count'] / 10
            del tmp3['last_ten_day_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_ten_day_buyfreq'].isnull(), 'last_ten_day_buyfreq'] = 0

            # 最后 20 天的购买次数  last_twoten_day_count
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-31')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-10-11')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_twoten_day_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_twoten_day_count'].isnull(), 'last_twoten_day_count'] = 0
            # 最后 20 的购买频率（ 次数 / 20 ）
            tmp3['last_twoten_day_buyfreq'] = tmp3['last_twoten_day_count'] / 20
            del tmp3['last_twoten_day_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_twoten_day_buyfreq'].isnull(), 'last_twoten_day_buyfreq'] = 0

            # 最后二个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-31')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-09-02')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_two_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_two_month_count'].isnull(), 'last_two_month_count'] = 0

            # 最后三个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-10-31')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2013-08-02')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'last_three_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_three_month_count'].isnull(), 'last_three_month_count'] = 0

            # 最后三个月的购买频率（次数 / 90）
            tmp3['last_three_month_buyfreq'] = tmp3['last_three_month_count'] / 90
            del tmp3['last_three_month_count']
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['last_three_month_buyfreq'].isnull(), 'last_three_month_buyfreq'] = 0

            # 第一个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2012-12-01')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2012-11-01')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'first_one_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['first_one_month_count'].isnull(), 'first_one_month_count'] = 0

            # 前二个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2012-12-31')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2012-11-01')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'first_two_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['first_two_month_count'].isnull(), 'first_two_month_count'] = 0

            # 前三个月的购买次数
            tmp1 = data[((data['order_pay_time'].dt.date).astype(str) <= '2013-01-29')]
            tmp2 = tmp1[((tmp1['order_pay_time'].dt.date).astype(str) >= '2012-11-01')]
            tmp2['count'] = 1
            tmp3 = tmp2.groupby(['customer_id'])['count'].agg({'first_three_month_count': 'sum'})
            customer_all = customer_all.merge(tmp3, on=['customer_id'], how='left')
            customer_all.loc[customer_all['first_three_month_count'].isnull(), 'first_three_month_count'] = 0

        if idx == 0:
            last_data = last_data.merge(customer_all, on='customer_id', how='left')
        else:
            all_data = all_data.merge(customer_all, on='customer_id', how='left')

    '''
       六轮：（价格） 4个
       每个顾客每个月的父订单的商品总金额（平均）     order_amount_month_sum
       每个顾客每个月的父订单的实付金额（平均）       order_total_payment_month_sum
       每个顾客每个月的父订单的优惠金额（平均）       order_total_discount_month_sum
       每个顾客购买的商品id总数量（种类）             goods_id_kinds_count
       '''
    for idx, data in enumerate([train_last, train_all]):
        customer_all = pd.DataFrame(data[['customer_id']]).drop_duplicates(['customer_id']).dropna()
        data['order_pay_time'] = pd.to_datetime(data['order_pay_time'], format="%Y/%m/%d")
        data['count'] = 1
        if idx == 0:
            month = 11
        else:
            month = 12
        # 每个顾客每个月的父订单的商品总金额（平均）
        tmp = data.groupby(['customer_id'])['order_amount'].agg({'order_amount_month_sum': 'sum'})
        tmp['order_amount_month_sum'] = tmp['order_amount_month_sum'] / month
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # ***(问题)*** 每个顾客购买的商品id总数量（种类）
        tmp = data.groupby(['customer_id'])['goods_id'].agg({'goods_id_kinds_count': 'count'})
        # tmp['goods_id_count'] = tmp['goods_id_count'] / tmp1['goods_id_count_div']
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 每个顾客每个月的父订单的实付金额（平均）
        tmp = data.groupby(['customer_id'])['order_total_payment'].agg({'order_total_payment_month_sum': 'sum'})
        tmp['order_total_payment_month_sum'] = tmp['order_total_payment_month_sum'] / month
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        # 每个顾客每个月的父订单的优惠金额（平均）
        tmp = data.groupby(['customer_id'])['order_total_discount'].agg({'order_total_discount_month_sum': 'sum'})
        tmp['order_total_discount_month_sum'] = tmp['order_total_discount_month_sum'] / month
        customer_all = customer_all.merge(tmp, on=['customer_id'], how='left')

        if idx == 0:
            last_data = last_data.merge(customer_all, on='customer_id', how='left')
        else:
            all_data = all_data.merge(customer_all, on='customer_id', how='left')

    return last_data, all_data

# 确定label值，由后30天中的数据，确定10.1号之前的所有顾客是否在后续30天继续购买
def generate_label(data,label):
    data['label'] = 0
    valid_idx_list = list(label['customer_id'].unique())
    data['label'][data['customer_id'].isin(valid_idx_list)] = 1
    return data
# 由 loss 值计算score
def re_loglossv(labels,preds):
    deta = 3.45
    y_true = labels   # you can try this eval metric for fun
    y_pred = preds
    p = np.clip(y_pred, 1e-10, 1-1e-10)
    loss = -1/len(y_true) * np.sum(y_true * np.log(p) * deta + (1 - y_true) * np.log(1-p))
    return 're_logloss',loss,False

def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    deta = 34
    p = np.clip(preds, 1e-10, 1-1e-10)
    score = -1/len(label) * np.sum(label * np.log(p) * deta + (1 - label) * np.log(1-p))  # mean_squared_error(label,preds)
    return 'myFeval',(score-0.2)


if __name__ =="__main__":
    train = pd.read_csv(r'G:\三枪一炮\YunJifen\second_contest\data2\round2_diac2019_train.csv',
                        parse_dates=['order_pay_time', 'goods_list_time', 'goods_delist_time'])
    train = pd.DataFrame(train)
    # 删除-- 双11
    # train= train[~(train['order_pay_time'].dt.date).astype(str).str.contains('2013-11-11')]

    train.loc[train['order_detail_amount'].isnull(), 'order_detail_amount'] = train['order_detail_discount'] + train[
        'order_detail_payment']

# ------------------------------------------------------------------------------------------
    # min:  2012-11-01      max:    2013-10-31
    train_last = train[((train['order_pay_time'].dt.date).astype(str) <= '2013-10-01')]
    # train_label = train[(train['order_pay_time'].dt.date).astype(str) >= '2013-09-02']
    train_label = train[(train['order_pay_time'].dt.date).astype(str) >= '2013-10-02']
    # 删除双十一数据
    #train_label= train_label[~(train_label['order_pay_time'].dt.date).astype(str).str.contains('2013-11-11')]

    train_all = train[((train['order_pay_time'].dt.date).astype(str) <= '2013-10-31')]
    print('train_last shape:', train_last.shape, 'train_label shape:', train_label.shape, 'train_all shape:',
          train_all.shape)

    # 去重复的customer_id
    last_data = pd.DataFrame(train_last[['customer_id']]).drop_duplicates(['customer_id']).dropna()
    all_data = pd.DataFrame(train_all[['customer_id']]).drop_duplicates(['customer_id']).dropna()

    train_last['order_pay_month'] = train_last['order_pay_time'].dt.month
    train_last['order_pay_dayofweek'] = train_last['order_pay_time'].dt.dayofweek
    train_last['order_pay_day'] = train_last['order_pay_time'].dt.day
    train_last['order_pay_dayofyear'] = train_last['order_pay_time'].dt.dayofyear

    train_all['order_pay_month'] = train_last['order_pay_time'].dt.month
    train_all['order_pay_dayofweek'] = train_last['order_pay_time'].dt.dayofweek
    train_all['order_pay_day'] = train_last['order_pay_time'].dt.day
    train_all['order_pay_dayofyear'] = train_all['order_pay_time'].dt.dayofyear

# --------------------------------------------------------------------------------
    train_last = train_last.reset_index()
    train_all = train_all.reset_index()

    # 每个用户的订单的时间差 order_pay_gap
    train_last = train_last.sort_values(by=['customer_id', 'order_pay_dayofyear'])
    train_last['order_pay_dayofyear_last'] = train_last['order_pay_dayofyear'].groupby(train_last['customer_id']).shift(
        1)
    train_last['order_pay_gap'] = train_last['order_pay_dayofyear'] - train_last['order_pay_dayofyear_last']
    train_last.loc[train_last['order_pay_gap'].isnull(), 'order_pay_gap'] = 0
    train_last = train_last.sort_values(by=['index'])

    train_all = train_all.sort_values(by=['customer_id', 'order_pay_dayofyear'])
    train_all['order_pay_dayofyear_last'] = train_all['order_pay_dayofyear'].groupby(train_all['customer_id']).shift(1)
    train_all['order_pay_gap'] = train_all['order_pay_dayofyear'] - train_all['order_pay_dayofyear_last']
    train_all.loc[train_all['order_pay_gap'].isnull(), 'order_pay_gap'] = 0
    train_all = train_all.sort_values(by=['index'])

#---------------------------------------------------------------------------------------------------
    last_data, all_data = Computer_features(train_last, train_all,last_data,all_data)


    for data in [last_data, all_data]:
        data['customer_city'] = LabelEncoder().fit_transform(data['customer_city'].fillna('None'))
        data['customer_province'] = LabelEncoder().fit_transform(data['customer_province'].fillna('None'))
    last_data = generate_label(last_data, train_label)
    # 最后一个外加指标: 最早---最晚，时间间隔
    last_data['order_pay_dayofyear_gap'] = last_data['order_pay_dayofyear_max'] - last_data['order_pay_dayofyear_min']
    all_data['order_pay_dayofyear_gap'] = all_data['order_pay_dayofyear_max'] - all_data['order_pay_dayofyear_min']
    print('-------------构建特征完成----------------------------------')
    # last_data.to_csv('./last_data2')
    # all_data.to_csv('./all_data2')

# 数据处理
    last = last_data
    all = all_data
    # 异常值填充 ，求众数,填充
    nums = all['order_pay_day_mean']  # 11
    mode = sorted(nums)[len(nums) // 2]
    all.loc[all['order_pay_day_mean'].isnull(), 'order_pay_day_mean'] = 11.0

    nums = all['order_pay_day_max']  # 15.0
    mode = sorted(nums)[len(nums) // 2]
    all.loc[all['order_pay_day_max'].isnull(), 'order_pay_day_max'] = 15.0

    nums = all['order_amount_month_min']  # 1.0
    mode = sorted(nums)[len(nums) // 2]
    all.loc[all['order_amount_month_min'].isnull(), 'order_amount_month_min'] = 1.0

    nums = all['order_amount_month_max']  # 1.0
    mode = sorted(nums)[len(nums) // 2]
    all.loc[all['order_amount_month_max'].isnull(), 'order_amount_month_max'] = 1.0

    all.loc[all['detail_discount_div_amount'] < 0, 'detail_discount_div_amount'] = 0
    all.loc[all['goods_status_2'] < 0, 'goods_status_2'] = 0  # 特殊异常值，商品状态编码为 众数代替
    last.loc[last['detail_discount_div_amount'] < 0, 'detail_discount_div_amount'] = 0
    last.loc[last['goods_status_2'] < 0, 'goods_status_2'] = 0  # 特殊异常值，商品状态编码为 众数代替

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

    # 添加修改 特征，
    #  < 0.8 counts , > 0.8 counts
    last['customer_counts_ll'] = 1
    last.loc[last['customer_counts'] < 0.8, 'customer_counts_ll'] = 0
    last.loc[last['is_member_actived'] > 0, 'is_member_actived'] = 1

    all['customer_counts_ll'] = 1
    all.loc[all['customer_counts'] < 0.8, 'customer_counts_ll'] = 0
    all.loc[all['is_member_actived'] > 0, 'is_member_actived'] = 1

    # 解决倾斜特征
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in last.columns:
        if last[i].dtype in numeric_dtypes:
            numeric.append(i)

    # 为所有特征绘制箱型线图
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_style('white')
    f, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale('log')
    ax = sns.boxplot(data=last[numeric], orient='h', palette='Set1')
    ax.xaxis.grid(False)
    ax.set(ylabel='Feature names')
    ax.set(xlabel='Numeric values')
    ax.set(title='Numeric Distribution of Features')
    sns.despine(trim=True, left=True)

    # 寻找偏弱的特征
    from scipy.stats import skew, norm

    skew_features = last[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index
    skewness = pd.DataFrame({'Skew': high_skew})
    skew_features.head(10)

    # 用scipy函数boxcox1p来计算Box-Cox转换。我们的目标是找到一个简单的转换方式使数据规范化。
    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax

    for i in skew_index:
        last[i] = boxcox1p(last[i], boxcox_normmax(last[i] + 1))

    # 处理所有的 skewed values
    sns.set_style('white')
    f, ax = plt.subplots(figsize=(8, 7))
    ax.set_xscale('log')
    ax = sns.boxplot(data=last[skew_index], orient='h', palette='Set1')
    ax.xaxis.grid(False)
    ax.set(ylabel='Feature names')
    ax.set(xlabel='Numeric values')
    # --------------------------------------------------------------------------------------------------

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in all.columns:
        if all[i].dtype in numeric_dtypes:
            numeric.append(i)

    # 寻找偏弱的特征
    from scipy.stats import skew, norm

    skew_features = all[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skew_features[skew_features > 0.5]
    skew_index = high_skew.index
    skewness = pd.DataFrame({'Skew': high_skew})
    skew_features.head(10)

    # 用scipy函数boxcox1p来计算Box-Cox转换。我们的目标是找到一个简单的转换方式使数据规范化。
    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax

    for i in skew_index:
        all[i] = boxcox1p(all[i], boxcox_normmax(all[i] + 1))


    # 构造 logs 特征， squares 特征
    def logs(res, ls):
        m = res.shape[1]
        for l in ls:
            res = res.assign(newcol=pd.Series(np.log(1.01 + res[l])).values)
            res.columns.values[m] = l + '_log'
            m += 1
        return res


    log_features = ['customer_counts', 'goods_price_mean',
                    'order_amount_sum', 'order_total_payment_sum',
                    'order_detail_amount_sum', 'order_detail_payment_sum',
                    'order_pay_day_mean', 'order_amount_month_sum',
                    'goods_id_kinds_count', 'order_total_payment_month_sum']

    last_features = logs(last, log_features)
    all_features = logs(all, log_features)


    def squares(res, ls):
        m = res.shape[1]
        for l in ls:
            res = res.assign(newcol=pd.Series(res[l] * res[l]).values)
            res.columns.values[m] = l + '_sq'
            m += 1
        return res


    squared_features = ['customer_counts', 'goods_price_mean',
                        'order_total_discount_mean', 'order_pay_day_mean',
                        'last_one_month_count', 'last_one_month_buyfreq', 'last_ten_day_count',
                        'last_ten_day_buyfreq', 'last_twoten_day_count',
                        'last_twoten_day_buyfreq', 'last_two_month_count',
                        'last_three_month_count', 'last_three_month_buyfreq']
    last_features = squares(last_features, squared_features)
    all_features = squares(all_features, squared_features)
#------------------------------------------------------------------------------------------------------------
    """
        全部： 1826575 个 customer_id
    """
    # last_data = pd.read_csv(r'G:\三枪一炮\YunJifen\last_data2_add1.csv')
    # all_data = pd.read_csv(r'G:\三枪一炮\YunJifen\all_data2_add1.csv')
    # 上采样
    # last_data = pd.read_csv(r'G:\三枪一炮\YunJifen\last_data_resampled1.csv')
    # 随机打乱
    from sklearn.utils import shuffle
    last_data= shuffle(last_data)
    all_data = shuffle(all_data)

    # last_data.sample(3000000)
    # last_data2 = last_data
    # 下采样，将 label = 0数据设置成 label=1 同样的数据量
    # last_data = lower_sample_data(last_data, percent=1)

# ----------------------------------------------------------------------------------------------------------------------------
    feature = ['customer_id', 'customer_counts',
       'customer_province', 'customer_city', 'long_time',
       'is_customer_rate_num', 'isnot_customer_rate_num', 'is_member_actived',
       'is_customer_have_discount_count', 'customer_gender_sum',
       'goods_price_max', 'goods_price_min', 'goods_price_mean',
       'goods_price_std', 'goods_has_discount_counts',
       'goods_hasnot_discount_counts', 'goods_status_2', 'goods_status_1',
       'order_amount_sum', 'order_total_payment_sum',
       'order_total_discount_mean', 'order_status_max',
       'order_detail_amount_sum', 'order_detail_payment_sum',
       'detail_discount_div_amount', 'order_detail_status_max',
       'order_pay_dayofyear_max', 'order_pay_dayofyear_min',
       'order_pay_rate_all', 'order_pay_rate_month', 'order_pay_rate_week',
       'order_pay_gap_max', 'order_pay_gap_min', 'order_amount_month_max',
       'order_amount_month_min', 'order_pay_day_max', 'order_pay_day_mean',
       'last_one_month_count', 'last_one_month_buyfreq', 'last_ten_day_count',
       'last_ten_day_buyfreq', 'last_twoten_day_count',
       'last_twoten_day_buyfreq', 'last_two_month_count',
       'last_three_month_count', 'last_three_month_buyfreq',
       'first_one_month_count', 'first_two_month_count',
       'first_three_month_count', 'order_amount_month_sum',
       'goods_id_kinds_count', 'order_total_payment_month_sum',
       'order_total_discount_month_sum', 'order_pay_dayofyear_gap',
       'customer_counts_log', 'goods_price_mean_log', 'order_amount_sum_log',
       'order_total_payment_sum_log', 'order_detail_amount_sum_log',
       'order_detail_payment_sum_log', 'order_pay_day_mean_log',
       'order_amount_month_sum_log', 'goods_id_kinds_count_log',
       'order_total_payment_month_sum_log', 'customer_counts_sq',
       'goods_price_mean_sq', 'order_total_discount_mean_sq',
       'order_pay_day_mean_sq', 'last_one_month_count_sq',
       'last_one_month_buyfreq_sq', 'last_ten_day_count_sq',
       'last_ten_day_buyfreq_sq', 'last_twoten_day_count_sq',
       'last_twoten_day_buyfreq_sq', 'last_two_month_count_sq',
       'last_three_month_count_sq', 'last_three_month_buyfreq_sq','label']
# ------------------------------------------------------------------------------------

    y = last_data['label']
    feature.pop(feature.index('label')) # 删除 label
    X = last_data[feature]
    X_all = all_data[feature]
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)


    import xgboost as xgb
    # xgb 模型
    from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    # xgb模型参数设置
    xgb_params = {"booster": 'gbtree',
                  'eta': 0.005,
                  'max_depth': 5,
                  'subsample': 0.7,
                  'colsample_bytree': 0.8,
                  'objective': 'binary:logistic',
                  'eval_metric': 'logloss',
                  'silent': True,
                  'nthread': 8,
                  'scale_pos_weight': 2.5   # 处理正负样本不均衡
                  }

# -----------------------------------------------------------------------------------------------------------

    oof_xgb = np.zeros(len(X_train))
    predictions_xgb = np.zeros(len(X_valid))
    watchlist = [(xgb.DMatrix(X_train.as_matrix(), y_train.as_matrix()), 'train'),
                 (xgb.DMatrix(X_valid.as_matrix(), y_valid.as_matrix()), 'valid_data')]
    clf = xgb.train(dtrain=xgb.DMatrix(np.array(X_train), np.array(y_train)), num_boost_round=500, evals=watchlist,
                    early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params, feval=myFeval)
    oof_xgb = clf.predict(xgb.DMatrix(X_valid.as_matrix()), ntree_limit=clf.best_ntree_limit)
    pred_xgb = clf.predict(xgb.DMatrix(X_all.as_matrix()), ntree_limit=clf.best_ntree_limit)
    res = all_data[['customer_id']]
    res['result'] = pred_xgb
    # 保存 xgb模型
    # clf.save_model('./xgb.model_true_false')
    # load model
    # bst2 = xgb.Booster(model_file='xgb.model1')

    data = pd.DataFrame(train[['customer_id']]).drop_duplicates(['customer_id']).dropna()
    data = (data.merge(res,on=['customer_id'],how='left')).sort_values(['customer_id'])
    data['customer_id'] = data['customer_id'].astype('int64')
    data['result'] = data['result'].fillna(0)
    result = data[['customer_id','result']]
    result.to_csv('./round2_diac2019_test.csv', index=False)


# ---------------------------------------------------------------------------------------------------------------

    train_x = X.as_matrix()
    train_y = y.as_matrix()
    oof_xgb = np.zeros(len(train_x))
    pred_xgb = np.zeros(len(X_all))
    # 五折交叉验证
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
        print("fold n°{}".format(fold_ + 1))
        trn_data = xgb.DMatrix(train_x[trn_idx], train_y[trn_idx])
        val_data = xgb.DMatrix(train_x[val_idx], train_y[val_idx])
        watchlist = [( trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=300, evals=watchlist, early_stopping_rounds=200,
                        verbose_eval=100, params=xgb_params, feval=myFeval)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(train_x[val_idx]), ntree_limit=clf.best_ntree_limit)
        pred_xgb += clf.predict(xgb.DMatrix(X_all.as_matrix()), ntree_limit=clf.best_ntree_limit) / folds.n_splits


    res = all_data[['customer_id']]
    res['result'] = pred_xgb
    print( "relogloss: ", re_loglossv(train_y, oof_xgb))
    data = pd.DataFrame(train[['customer_id']]).drop_duplicates(['customer_id']).dropna()
    data = (data.merge(res,on=['customer_id'],how='left')).sort_values(['customer_id'])
    data['customer_id'] = data['customer_id'].astype('int64')
    data['result'] = data['result'].fillna(0)
    result = data[['customer_id','result']]
    result.to_csv('./round2_diac2019_test.csv', index=False)


