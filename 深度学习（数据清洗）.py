import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#原始数据集
data_order = {
    'order_id': [2003, 2004, 2005, 2006, 2007, 2008],
    'customer_name': ['张三', '李四', '王五', None, '赵六', '钱七'],
    'city': ['北京', '北京', '广州', '深圳', None, '成都'],
    'age': [None, 30, 28, 35, 40, None],
    'order_amount': [1000, None, 1500, 2000, 9999, 800],
    'days_to_delivery': [None, 15, 20, 25, 99, 30],
    'is_member': [1, 0, 1, None, 1, 0]
}
df_raw = pd.DataFrame(data_order)
print(df_raw)
print(df_raw.describe(include="all"))
print(df_raw.isnull().sum())
#Z-score 检测订单金额异常
print("\n" + "-"*131)
order_amount_z = np.abs(stats.zscore(df_raw["order_amount"].dropna()))
print(order_amount_z)
print(f"Z-score最大值: {order_amount_z.max():.2f}")
#IQR异常值检测
print("\n" + "-"*131)
Q1 = df_raw['days_to_delivery'].quantile(0.25)
Q3 = df_raw['days_to_delivery'].quantile(0.75)
IQR = Q3 - Q1
l = Q1 - 1.5 * IQR
u = Q3 + 1.5 * IQR
print(df_raw[(df_raw['days_to_delivery'] < l) | (df_raw['days_to_delivery'] > u)])
print(df_raw.dtypes)
#复制原始数据集
df_clean = df_raw.copy()
# 数据清洗，直接赋值
df_clean['customer_name'] = df_clean['customer_name'].fillna("匿名用户")
city_mode = df_clean['city'].mode()[0]
df_clean['city'] = df_clean['city'].fillna(city_mode)
age_median = df_clean['age'].median()
df_clean['age'] = df_clean['age'].fillna(age_median)
#异常值修正，先转为nan
def clean_value(value, lower, upper):
    if pd.isnull(value):
        return value
    elif value > upper or value < lower:
        return np.nan
    else:
        return value
#填充中位数
df_clean['order_amount'] = df_clean['order_amount'].apply(lambda x: clean_value(x, 0, 2000))
df_clean['days_to_delivery'] = df_clean['days_to_delivery'].apply(lambda x: clean_value(x, 1, 30))
# 订单金额
order_amount_median = df_clean['order_amount'].median()
df_clean['order_amount'] = df_clean['order_amount'].fillna(order_amount_median)
# 配送天数
days_to_delivery_median = df_clean['days_to_delivery'].median()
df_clean['days_to_delivery'] = df_clean['days_to_delivery'].fillna(days_to_delivery_median)

is_member_mode = df_clean['is_member'].mode()[0]
df_clean['is_member'] = df_clean['is_member'].fillna(is_member_mode)

print(df_clean)
print(df_clean.isnull().sum())
#可视化视图
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
#子图1
axes[0, 0].boxplot([df_raw['order_amount'].dropna(), df_clean['order_amount']], tick_labels=['Raw', 'Clean'])
axes[0, 0].set_title('Order Amount: Raw vs Clean')
axes[0, 0].set_ylabel('Amount')
#子图2
axes[0, 1].boxplot([df_raw['days_to_delivery'].dropna(), df_clean['days_to_delivery']], tick_labels=['Raw', 'Clean'])
axes[0, 1].set_title('Days to Delivery: Raw vs Clean')
axes[0, 1].set_ylabel('Days')
#子图3
axes[1, 0].boxplot([df_raw['age'].dropna(), df_clean['age']], tick_labels=['Raw', 'Clean'])
axes[1, 0].set_title('Age: Raw vs Clean')
axes[1, 0].set_ylabel('Age')
#子图4
axes[1, 1].text(0.5, 0.5, 'Data Cleaning Complete\nNo missing values remain.',
               horizontalalignment='center',
               verticalalignment='center',
               transform=axes[1, 1].transAxes,
               fontsize=12)
axes[1, 1].axis('off')
axes[1, 1].set_title('Processing Status')
#显示图表
plt.tight_layout()
plt.show()