#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install plotly


# In[5]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# In[6]:


msft = yf.Ticker("MSFT") #get microsoft as trial
hist = msft.history(period = '5y', interval = '1d') #5 year data with 1 day as the interval
df = pd.DataFrame(hist) #make a dataframe for further process
df['Date'] = df.index
df.head()


# In[7]:


fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




