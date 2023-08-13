#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


train=pd.read_csv('titanic_train.csv')


# In[18]:


train.head()


# In[20]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[21]:


sns.set_style('whitegrid')


# In[23]:


sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[24]:


sns.countplot(x='Survived',hue='Pclass',data=train,palette='RdBu_r')


# In[26]:


sns.displot(train['Age'].dropna(),kde=False,bins=30)


# In[27]:


train['Age'].plot.hist(bins=35)


# In[28]:


train.info()


# In[29]:


sns.countplot(x='SibSp',data=train,)


# In[30]:


train['Fare'].hist(bins=40,figsize=(10,4))


# In[31]:


import cufflinks as cf


# In[32]:


cf.go_offline()


# In[33]:


train['Fare'].iplot(kind='hist',bins=30) #put bins=50 also and see what happens


# In[34]:


#CLEANING OUR DATA


# In[35]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[36]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)


# In[37]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
        
        


# In[38]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[39]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[40]:


#there is so much missing information in cabin column that it is just better to drop it


# In[41]:


train.drop('Cabin',axis=1,inplace=True)


# In[42]:


train.head()


# In[43]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[44]:


train.dropna(inplace=True)#just to drop one row of Embarked


# In[45]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[47]:


pd.get_dummies(train['Sex'],drop_first=True) #drop_first because our machine will know that if a column is female(0)then male will be its exact opposite and a problem of multi linear collinearity arises and machine gets confused bw two columns so we drop the first 


# In[48]:


sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[49]:


sex.head()


# In[50]:


embark=pd.get_dummies(train['Embarked'],drop_first=True)#C will be dropped and Q and S are not perfect predictors of each other so no problem arises


# In[51]:


train=pd.concat([train,sex,embark],axis=1)


# In[52]:


train.head(2)


# In[53]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[54]:


train.head()


# In[55]:


#PassengerId is just an index and not actual official ID so we drop this column


# In[56]:


train.drop('PassengerId',axis=1,inplace=True)


# In[57]:


train.head()


# In[58]:


#Training the data


# In[59]:


X=train.drop('Survived',axis=1)
y=train['Survived']


# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[63]:


from sklearn.linear_model import LogisticRegression


# In[71]:


logmodel= LogisticRegression(solver='lbfgs',max_iter=1000)


# In[72]:


logmodel.fit(X_train,y_train)


# In[73]:


predictions=logmodel.predict(X_test)


# In[74]:


from sklearn.metrics import classification_report


# In[75]:


print(classification_report(y_test,predictions))


# In[76]:


from sklearn.metrics import confusion_matrix


# In[77]:


confusion_matrix(y_test,predictions)


# In[ ]:




