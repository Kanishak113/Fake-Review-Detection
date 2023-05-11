#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing Necessary libraries and packages


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#loading the dataset and analyzing it


# In[8]:


data = pd.read_csv('yelp.csv')
print("Shape of the dataset:")
print(data.shape)
print("Column names:")
print(data.columns)
print("Datatype of each column:")
print(data.dtypes)
print("Few dataset entries:")
print(data.head())
data.describe(include='all')


# In[ ]:


#creating a new coloumn for the number of words in the review


# In[9]:


data['length'] = data['text'].apply(len)
data.head()


# In[ ]:


#performing the visulaization


# In[10]:


graph = sns.FacetGrid(data=data,col='stars')
graph.map(plt.hist,'length',bins=50,color='blue')


# In[ ]:


#finding the mean value of vote columns


# In[11]:


stval = data.groupby('stars').mean()
stval


# In[ ]:


#finding correlation between the vote columns


# In[12]:


stval.corr()


# In[ ]:


#classifiying the dataset and splitting it in reviews and stars


# In[13]:


data_classes = data[(data['stars']==1) | (data['stars']==3) | (data['stars']==5)]
data_classes.head()
print(data_classes.shape)

x = data_classes['text']
y = data_classes['stars']
print(x.head())
print(y.head())


# In[ ]:


#data cleaning 


# In[14]:


def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


#vectorization 


# In[20]:


vocab = CountVectorizer(analyzer=text_process).fit(x)
print(len(vocab.vocabulary_))
r0 = x[0]
print(r0)
vocab0 = vocab.transform([r0])
print(vocab0)
print("Getting the words back:")
print(vocab.get_feature_names_out()[19648])
print(vocab.get_feature_names_out()[10643])


# In[ ]:


#vecotorization of whole review set and checking the sparse matrix


# In[21]:


x = vocab.transform(x)
print("Shape of the sparse matrix: ", x.shape)
print("Non-Zero occurences: ",x.nnz)

density = (x.nnz/(x.shape[0]*x.shape[1]))*100
print("Density of the matrix = ",density)


# In[ ]:


#splitting the dataset into testing and training set


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# In[ ]:


#modeling


# In[ ]:


#Multinomial Naive Bayes


# In[24]:


from sklearn.naive_bayes import MultinomialNB
model1 = MultinomialNB()
model1.fit(x_train,y_train)
pred = model1.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,pred))
print("Score:",round(accuracy_score(y_test,pred)*100,2))
print("Classification Report:",classification_report(y_test,pred))


# In[ ]:


#Random Forest Classifier


# In[25]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(x_train,y_train)
pred = model2.predict(x_test)
print("Confusion Matrix for Random Forest Classifier:")
print(confusion_matrix(y_test,pred))
print("Score:",round(accuracy_score(y_test,pred)*100,2))
print("Classification Report:",classification_report(y_test,pred))


# In[ ]:


#Decision Tree


# In[26]:


from sklearn.tree import DecisionTreeClassifier
model3= DecisionTreeClassifier()
model3.fit(x_train,y_train)
pred = model3.predict(x_test)
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test,pred))
print("Score:",round(accuracy_score(y_test,pred)*100,2))
print("Classification Report:",classification_report(y_test,pred))


# In[ ]:


#Support Vector Machines


# In[27]:


from sklearn.svm import SVC
model4 = SVC(random_state=101)
model4.fit(x_train,y_train)
pred = model4.predict(x_test)
print("Confusion Matrix for Support Vector Machines:")
print(confusion_matrix(y_test,pred))
print("Score:",round(accuracy_score(y_test,pred)*100,2))
print("Classification Report:",classification_report(y_test,pred))


# In[ ]:


#Gradient Boosting Classifier


# In[28]:


from sklearn.ensemble import GradientBoostingClassifier
model5 = GradientBoostingClassifier(learning_rate=0.1,max_depth=5,max_features=0.5,random_state=999999)
model5.fit(x_train,y_train)
pred = model5.predict(x_test)
print("Confusion Matrix for Gradient Boosting Classifier:")
print(confusion_matrix(y_test,pred))
print("Score:",round(accuracy_score(y_test,pred)*100,2))
print("Classification Report:",classification_report(y_test,pred))


# In[ ]:


#K - Nearest Neighbor Classifier


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
model6 = KNeighborsClassifier(n_neighbors=10)
model6.fit(x_train,y_train)
pred = model6.predict(x_test)
print("Confusion Matrix for K Neighbors Classifier:")
print(confusion_matrix(y_test,pred))
print("Score: ",round(accuracy_score(y_test,pred)*100,2))
print("Classification Report:")
print(classification_report(y_test,pred))


# In[ ]:


#Score Comparision Visualization 


# In[57]:


scorenb=round(model1.score(x_test,y_test)*100,2)
scorerf=round(model2.score(x_test,y_test)*100,2)
scoredt=round(model3.score(x_test,y_test)*100,2)
scoresvm=round(model4.score(x_test,y_test)*100,2)
scorexgb=round(model5.score(x_test,y_test)*100,2)
scoreknn=round(model6.score(x_test,y_test)*100,2)
print("Scores are as follows:")
print("Multinomial Naive Bayes : "+str(scorenb))
print("Random Forest Classifier : "+str(scorerf))
print("Decision Tree : "+str(scoredt))
print("Support Vector Machines : "+str(scoresvm))
print("Gradient Boosting Classifier : "+str(scorexgb))
print("K Neighbors Classifier : "+str(scoreknn))


# In[ ]:


#creating graph to visualize better


# In[41]:


import matplotlib.pyplot as plt


# In[44]:


models=['NB','RF','DT','SVM','XGB','KNN']
accuracy=[scorenb,scorerf,scoredt,scoresvm,scorexgb,scoreknn]


# In[47]:


plt.plot(models,accuracy,'b-o',label='Accuracy comparision plot');
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[50]:


# POSITIVE REVIEW
pr = data['text'][0]
print(pr)
print("Actual Rating: ",data['stars'][0])
pr_t = vocab.transform([pr])
print("Predicted Rating:")
model1.predict(pr_t)[0]


# In[51]:


#Average Review
ar = data['text'][16]
print(ar)
print("Actual Rating: ",data['stars'][16])
ar_t = vocab.transform([ar])
print("Predicted Rating:")
model1.predict(ar_t)[0]


# In[52]:


#Negative Review
nr = data['text'][16]
print(nr)
print("Actual Rating: ",data['stars'][23])
nr_t = vocab.transform([nr])
print("Predicted Rating:")
model1.predict(nr_t)[0]


# In[53]:


count = data['stars'].value_counts()
print(count)

