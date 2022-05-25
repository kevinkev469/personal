#!/usr/bin/env python
# coding: utf-8

# In[25]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = True')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[7]:


import nltk


# In[8]:


import sklearn


# In[16]:


from sklearn.datasets import load_boston


# In[17]:


boston = load_boston()


# In[18]:


print(boston.DESCR)


# In[ ]:





# In[19]:


plt.hist(boston.target, bins=30)
plt.xlabel("Price in $1000s")
plt.ylabel("Number of Houses")


# In[8]:


plt.scatter(boston.data[:,5], boston.target)
plt.xlabel("Average no. of rooms")
plt.ylabel("Price in '000 USD")


# In[20]:


boston_df = pd.DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df["Price"] = boston.target
boston_df.head()


# In[26]:


sns.lmplot("RM", "Price", boston_df)


# In[28]:


x = boston.data[:, np.newaxis,5]
y = boston.target


# In[29]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x,y)


# In[24]:


linreg.score(x,y) #Multiple R-Sqaure


# In[25]:


linreg.intercept_


# In[17]:


linreg.coef_


# In[34]:


#Multiple linear regression

mlinreg = LinearRegression()


# In[39]:


X = boston.data # X in caps
Y = boston.target # Y in caps


# In[35]:


mlinreg.fit(X,Y)


# In[37]:


mlinreg.score(X,Y)


# In[38]:


mlinreg.intercept_


# In[40]:


mlinreg.coef_


# In[41]:


boston.data    #there is a difference between boston and boston.data

MACHINE LEARNING
# In[5]:


from sklearn.model_selection import train_test_split


# In[21]:


np.random.seed(2021)
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size=.8)


# In[23]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape )


# In[30]:


linreg = LinearRegression()


# In[31]:


linreg.fit(x_train, y_train)    #impact of the 13 variables on price


# In[33]:


y_pred = linreg.predict(x_test)
y_pred


# In[34]:


y_test


# In[35]:


np.mean((y_test - y_pred)**2) #mean squared error


# In[36]:


from sklearn.metrics import mean_squared_error


# In[37]:


mean_squared_error(y_test, y_pred)


# In[38]:


(mean_squared_error(y_test, y_pred))**(1/2)  #ROOT MEAN SQUARE ERROR


# In[ ]:





# In[ ]:




