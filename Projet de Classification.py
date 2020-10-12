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
from sklearn import neighbors
model=neighbors.KNeighborsClassifier(n_neighbors=2)


# In[25]:


sample=np.random.randint(mnist.data.shape[0],size=int(mnist.data.shape[0]*0.08))
sample


# In[26]:


data=mnist.data[sample]
target=mnist.target[sample]
x_train,x_test,y_train,y_test=tts(data,target,train_size=0.8)


# In[27]:


model.fit(x_train,y_train)


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
