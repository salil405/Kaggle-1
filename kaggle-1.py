#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[2]:


feature_data=pd.read_csv(r'C:\Users\salil\Downloads\traindata - traindata.csv',converters={'features':eval})
feature_data


# In[3]:


target_data=pd.read_csv(r'C:\Users\salil\Downloads\trainlabel - trainlabel.csv')


# In[4]:


test_data=pd.read_csv(r'C:\Users\salil\Downloads\testdata - testdata.csv',converters={'features':eval})


# In[5]:


x_train_1= pd.DataFrame(feature_data['features'].tolist())
y_train_1= target_data.iloc[:,1]
x_test_1= pd.DataFrame(test_data['features'].tolist())


# In[6]:


x_train_1


# In[7]:


y_train_1


# In[8]:


x_test_1


# In[9]:


#from sklearn import preprocessing
#scaler = preprocessing.MinMaxScaler()
#x_train_2 = scaler.fit_transform(x_train_1)
#x_train_2


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_train_1,y_train_1,test_size=0.15)


# In[14]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(x_train, y_train)
dtree_predictions = dtree_model.predict(x_test)
accuracy=dtree_model.score(x_test,y_test)
print(accuracy)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1).fit(x_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(x_test, y_test)
print(accuracy)


# In[15]:


# training a SVM classifier
from sklearn.svm import SVC
svm_model_poly = SVC(kernel = 'poly', degree = 2,random_state=1, class_weight='balanced').fit(x_train, y_train)
svm_predictions = svm_model_poly.predict(x_test)

# model accuracy for X_test
accuracy_1 = svm_model_poly.score(x_test, y_test)
print(accuracy_1)


# In[13]:


# training a  SVM classifier
from sklearn.svm import SVC
svm_model_rbf = SVC(kernel = 'rbf', C=2.85,random_state=1, class_weight='balanced').fit(x_train, y_train)
svm_predictions = svm_model_rbf.predict(x_test)

# model accuracy for X_test
accuracy_2 = svm_model_rbf.score(x_test, y_test)
print(accuracy_2)
# creating a confusion matrix
#cm = confusion_matrix(y_test, svm_predictions)

