#!/usr/bin/env python
# coding: utf-8

# In[1]


from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as pl


# In[2]:


mnist = fetch_openml('mnist_784', version=1)


# In[24]:


from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearch

sample=np.random.randint(mnist.data.shape[0],size=int(mnist.data.shape[0]*0.08))
sample


# In[26]:


data=mnist.data[sample]
target=mnist.target[sample]
x_train,x_test,y_train,y_test=tts(data,target,train_size=0.8)





from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklear.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

Classifiers=[KNeighborsClassifier(),SVC(),ogisticRegression(),RandomForestClassifier(),RandomForestClassifier(),MLPClassifier(),MLPClassifier(),
             LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),XGBClassifier()]


# In[25]:

import pandas as pd

col=["Name","Accuracy"]
df=pd.DataFrame(columns=col)
from classifier in Classifiers:
    Name=classifier.__name__
    model=classier.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    print("***Result***")
    print("="*40)
    print("{} \n Accuracy:{}".format(Name,acc))
    print("="*40)
    df1=pd.DataFrame([[Name,acc]],columns=col)
    df=df.append(df1)

# In[32]:


print("L'erreur sur prediction est de :{}$%$ ".format(((1-model.score(x_train,y_train))*100)))


# In[29]:


predicted=model.predict(x_test)


# In[30]:


images=x_test.reshape(-1,28,28)
select=np.random.randint(images.shape[0],size=20)


# In[31]:


fig,ax=pl.subplots(4,5)
for index,values in enumerate(select):
    pl.subplot(4,5,index+1)
    pl.imshow(images[values])
    pl.title(predicted[values])
    pl.axis('off')
pl.show()
print("L'erreur de prediction: ",1-model.score(x_train,y_train))


# In[21]:


err=[]
for i in range(2,30):
    model1=neighbors.KNeighborsClassifier(n_neighbors=i)
    model1.fit(x_train,y_train)
    err.append(1-model1.score(x_train,y_train))


# In[22]:


pl.plot(range(2,30),err)
