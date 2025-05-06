#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


#

# In[ ]:





# ### pd

# In[2]:


# 样本数量
n_samples = 1000
# 特征数量
n_features = 10
# 生成 ID 列
ids = np.arange(1, n_samples + 1)
# 生成 20 个特征列
features = np.random.randn(n_samples, n_features)
# 生成二元干预列
treatment = np.random.randint(0, 2, n_samples)
# 生成二元标签列
label = np.random.randint(0, 2, n_samples)
# 创建 DataFrame
columns = ['id'] + [f'feature_{i + 1}' for i in range(n_features)] + ['treatment', 'label']
data = np.hstack((ids.reshape(-1, 1), features, treatment.reshape(-1, 1), label.reshape(-1, 1)))
df1 = pd.DataFrame(data, columns=columns)


# In[22]:


df1.head(1)


# In[5]:


df2 = df1.copy(deep=True)
df2.rename(columns={f'feature_{i}': f'feature_{i+10}' for i in range(1, 11)}, inplace=True)
df2.drop(columns=['treatment', 'label'], inplace=True)


# In[21]:


df2.head(1)


# In[10]:


# 排序, 默认ascending=True升序排列
df1.sort_values(by=['feature_1', 'feature_2'], ascending=[True, True], inplace=True)


# In[11]:


# 合并
df = df1.merge(df2, on=['id'], how='left')


# In[15]:


# 拼接, 默认axis=0列对齐拼接，需要列名相同， axis=1行对齐拼接
common_cols = ['id']
df3 = pd.concat([df1[common_cols], df2[common_cols]], axis=0, ignore_index=True)
df3.columns


# In[20]:


# 去重
# df1.drop_duplicates(subset=['feature_1', 'feature_2'], keep='first', inplace=False)


# In[19]:


# 索引重制
df1.reset_index(drop=True, inplace=True)


# In[50]:


# df1.set_index(['id'])['feature_1'].to_dict()


# In[47]:


df1['new_id'] = df1.id.apply(lambda x: x if x > 0 else abs(x))


# In[ ]:





# ### 日期

# In[23]:


from datetime import datetime, timedelta


# In[35]:


today_date = datetime.now().date()
yesterday_date = (today_date - timedelta(days=1)).strftime("%Y-%m-%d")
today_date = today_date.strftime("%Y-%m-%d")
today_date


# In[38]:


# date_str 字符串格式: date_obj: 日期格式
# 日期格式转为字符串格式: date_obj.strftime("%Y-%m-%d")
# 字符串格式转为日期格式: datetime.strptime(date_str, "%Y-%m-%d")
# type(date_obj)


# In[41]:


# timedelta
today_date = datetime.now().date()
today_date + timedelta(days=1)


# In[44]:


# date_range
[str(dt)[:10] for dt in pd.date_range(start='2025-01-01', end='2025-01-03')]


# In[45]:


# to_datetime: 转为日期格式
# 单个字符串转换
date_str = '2024-05-10'
date = pd.to_datetime(date_str)
print(date)

# 字符串列表转换
date_str_list = ['2024-05-10', '2024-05-11', '2024-05-12']
dates = pd.to_datetime(date_str_list)
print(dates)


# In[ ]:





# In[ ]:





# In[49]:


try:
    get_ipython().system('jupyter nbconvert --to python df.ipynb')
    # file_name.ipynb是当前模块的文件名
    # 转化完成后会生成file_name.py
except:
    pass


# In[ ]:




