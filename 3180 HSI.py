#!/usr/bin/env python
# coding: utf-8

# X Allow shorting

# In[2]:


pip install pandas_datareader


# In[2]:


import pandas_datareader as pdr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data_index = ['0883.HK','0267.HK','0960.HK','1038.HK','0006.HK','0003.HK','2628.HK','0241.HK','1044.HK','0012.HK',
              '0002.HK','0101.HK','2020.HK','1109.HK','0016.HK','2018.HK','0017.HK','1093.HK','2269.HK','1997.HK',
              '1810.HK','0386.HK','9988.HK','2319.HK','1299.HK','0669.HK','1928.HK','0027.HK','1398.HK','3690.HK']
stockdf = pd.DataFrame()
dailydf = pd.DataFrame()
for i in data_index:
    HKEX = pdr.get_data_yahoo(i, start = "2019-12-1", end = "2021-5-1")
    daily_returns = HKEX['Adj Close'].resample('D').ffill().pct_change()
    stockdf[i] = daily_returns


# In[4]:


stock = stockdf.iloc[1:]
sigmaa = stock.cov()
mu = np.mean(stock,axis=0)


# In[5]:


DJI = pdr.get_data_yahoo('^HSI',start = '2019-12-1',end = '2021-5-1')
dji = DJI['Adj Close']
fig = plt.gcf()
fig.set_size_inches(12, 7.5)
plt.plot(dji,color='black')
plt.title('Covid impact on Hang Seng Index')
plt.show()


# In[6]:


import cvxpy as cp
def GMVP(Sigma):
    w = cp.Variable(len(mu))
    variance = cp.quad_form(w,Sigma)
    problem = cp.Problem(cp.Minimize(variance)
                         , [w>=0,cp.sum(w)==1])
    problem.solve()
    return w.value


# In[14]:


w_gmvp = GMVP(sigmaa)
w_gmvp


# In[24]:


def MVP(mu,Sigma,lmd = -1):
    w = cp.Variable(len(mu))
    variance = cp.quad_form(w,Sigma)
    expected_return = w@mu
    obj = variance - lmd*expected_return
    problem = cp.Problem(cp.Minimize(obj),[w>=0,cp.sum(w)==1])
    problem.solve()
    return w.value


# In[25]:


w_mvp = MVP(mu,sigmaa)


# In[17]:


def MSRP(mu,Sigma):
    w = cp.Variable(len(mu))
    variance = cp.quad_form(w,Sigma)
    problem = cp.Problem(cp.Minimize(variance),[w>=0,w@mu==1])
    problem.solve()
    return w.value/np.sum(w.value)


# In[18]:


w_msrp = MSRP(mu,sigmaa) 
w_msrp


# In[19]:


w_mdp = MSRP(np.sqrt(np.diag(sigmaa)),sigmaa)
w_mdp


# In[20]:


port_name = ['GMVP','MVP','MSRP','MDP']
data_indexs = ['0883','0267','0960','1038','0006','0003','2628','0241','1044','0012',
              '0002','0101','2020','1109','0016','2018','0017','1093','2269','1997',
              '1810','0386','9988','2319','1299','0669','1928','0027','1398','3690']
w_all = dict(weights = np.concatenate([w_gmvp, w_mvp, w_msrp, w_mdp]),
            stocks = 4*data_indexs, 
            portfolio = np.repeat(port_name,len(data_index)))
df = pd.DataFrame(w_all)
fig = plt.gcf()
fig.set_size_inches(20, 10)
ax = sns.barplot(x='stocks', y='weights', hue='portfolio', data=df)
plt.title('Portfolio allocation',size=20)
ax.legend(loc='best',prop={'size': 10})
plt.draw()


# In[26]:


ret_gmvp = stock@w_gmvp
ret_mvp = stock@w_mvp
ret_msrp = stock@w_msrp
ret_mdp = stock@w_mdp
fig = plt.gcf()
ax = plt.gca()
fig.set_size_inches(10, 8)
plt.plot(1 + np.cumsum(ret_gmvp), label='GMVP')
plt.plot(1 + np.cumsum(ret_mvp), label='MVP')
plt.plot(1 + np.cumsum(ret_msrp), label='MSRP')
plt.plot(1 + np.cumsum(ret_mdp), label='MDP')
plt.legend()
plt.title('Performance that not compounded')
plt.show()


# In[22]:


fig = plt.gcf()
ax = plt.gca()
fig.set_size_inches(10, 8)
plt.plot(np.cumprod(1 + ret_gmvp), label='GMVP')
plt.plot(np.cumprod(1 + ret_mvp), label='MVP')
plt.plot(np.cumprod(1 + ret_msrp), label='MSRP')
plt.plot(np.cumprod(1 + ret_mdp), label='MDP')
plt.legend()
plt.title('Performance that compounded')
plt.show()


# In[27]:


x1 = (1+ret_gmvp).cumprod()
p1 = x1.cummax()
d1 = (x1-p1)/p1
plt.plot(d1,label='GMVP')
x2 = (1+ret_mvp).cumprod()
p2 = x2.cummax()
d2 = (x2-p2)/p2
plt.plot(d2,label='MVP')
x3 = (1+ret_msrp).cumprod()
p3 = x3.cummax()
d3 = (x3-p3)/p3
plt.plot(d3,label='MSRP')
x4 = (1+ret_mdp).cumprod()
p4 = x4.cummax()
d4 = (x4-p4)/p4
plt.plot(d4,label='MDP')
plt.legend()
plt.title('Drawdown of Different Portfolio')
fig = plt.gcf()
ax = plt.gca()
fig.set_size_inches(10, 6)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




