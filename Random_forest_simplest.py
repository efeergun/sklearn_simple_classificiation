
# coding: utf-8

# # Hello 

# We firstly should import our Classifier and the data.
# 
# Luckily, Scikit-learn has the iris dataset.

# In[1]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


# we get data with the command load_iris().

# In[2]:


iris_data = load_iris()


# we divide our training data as "data", and training labels as "labels"

# In[3]:


data = iris_data.data
labels = iris_data.target


# we set our RandomForestClassifier as "my_model" 

# In[4]:


my_model = RandomForestClassifier()          #since it is just a simple ML example, we leave our model as default.


# In[5]:


my_model.fit(data,labels)


# then we test our model's predictions with any value to check results.

# In[6]:


my_model.predict([[4.9,3.1,1.5,0.2]])

