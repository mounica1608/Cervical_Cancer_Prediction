#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sklearn 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r'C:\Users\mouni\Downloads\risk_factors_cervical_cancer.csv')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


def data_type(df):
    for i in df.columns:
        print(f'{i}: {df[i].dtypes}')


# In[7]:


data_type(df)


# In[8]:


for i in df.columns:
    df[i].replace("?",np.nan,inplace=True)


# In[9]:


df.head()


# #### Handling Missing Values 

# In[10]:


cols = []

for i in df.columns:
    num= df[i].isnull().sum()
    if  num != 0:
        cols.append(i)
            
            


# In[11]:


cols


# In[12]:



for i in df[cols]:
    num = df[i].isnull().sum()
    print(i)
    print(num)
    print('')


# In[13]:


df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'],axis=1,inplace=True)


# In[14]:


df.shape


# #### Handling Missing Values

# In[15]:


print(df.dtypes)


# In[16]:


df['Number of sexual partners'].value_counts(normalize=True,ascending=False)


# In[17]:


df['Number of sexual partners'].isnull().mean()


# In[18]:


df['Number of sexual partners'] =  df['Number of sexual partners'].astype(str).astype(float)


# In[19]:


for i in df.columns:
    df[i] =  df[i].astype(str).astype(float)


# In[20]:


print(df.dtypes)


# In[21]:


import seaborn as sns
sns.countplot(df['Number of sexual partners'])
plt.show()


# In[22]:


df['Number of sexual partners'].isnull().sum()


# In[23]:


def imputation(df):
    for i in  df.columns:
        num =df[i].mode(dropna=True).loc[0]
        df[i].fillna(num,inplace=True)
    
    


# In[24]:


imputation(df)


# In[25]:


df['Number of sexual partners'].isnull().sum()


# In[26]:


df.head()


# In[27]:


for i in df.columns:
    print(f'{i}: {df[i].isnull().sum()}')


# In[ ]:





# In[28]:


df['STDs:cervical condylomatosis'].value_counts()
df.drop('STDs:cervical condylomatosis', axis=1,inplace=True)


# In[66]:





# In[29]:


df['STDs:AIDS'].value_counts()


# In[30]:


df.drop('STDs:AIDS', axis=1,inplace=True)


# In[31]:


df.shape


# ### Feature Selection

# In[42]:


corr = df.corr()
plt.figure(figsize=(22,24))
sns.heatmap(corr,vmin=0,vmax=1,annot=True)


# ### Predictive Modelling
# 

# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


X = df.drop('Dx:Cancer',axis=1)
Y = df['Dx:Cancer']


# In[57]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)


# In[58]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
Y_pred = rf.predict(X_test)


# In[59]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[61]:


accuracy_score(Y_test, Y_pred)


# In[62]:


confusion_matrix(Y_test, Y_pred)


# In[64]:


print(classification_report(Y_test, Y_pred))


# In[65]:


Y_pred


# In[68]:


np.array(Y_test)


# ### Cross Validation

# In[89]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(X)
print(kf)


# In[90]:


accu = []
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    rf.fit(X_train,Y_train)
    Y_pred = rf.predict(X_test)
    accu.append(accuracy_score(Y_test, Y_pred))
    
    


# In[91]:


np.mean(accu)


# In[92]:


accu


# In[ ]:




