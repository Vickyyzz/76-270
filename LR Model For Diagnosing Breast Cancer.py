#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets

# Load the data
data_set = datasets.load_breast_cancer()

# Print out the feature names
print ('Feature names:')
print(data_set.feature_names)

print('\n')

# Print out the labels/classifications
print ('Classification outcomes:')
print(data_set.target_names)


# In[2]:


from sklearn.model_selection import train_test_split
# Features
X=data_set.data
# Label('malignant'/'benign')
y=data_set.target
# Split the data into training data and testing data by ratio 3:1
# Training data features: X_train, training data labels: y_train
# Testing data features: X_test, testing data labels: y_test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)


# In[3]:


from sklearn.preprocessing import StandardScaler
# Initialize a scaler for normalising input data
sc=StandardScaler()
# Transform the trainging data
X_train = sc.fit_transform(X_train)
# Transform the testing data
X_test = sc.transform(X_test)


# In[4]:


from sklearn.linear_model import LogisticRegression
# Initialize a classifier
classifier = LogisticRegression(random_state = 0)
# Fit the classifier
classifier.fit(X_train, y_train)


# In[5]:


y_pred = classifier.predict(X_test)
correct = (y_test == y_pred).sum()
accuracy = correct / len(y_pred) * 100
print ('Accuracy:')
print(accuracy)


# In[ ]:




