#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib 


# In[3]:


import joblib 
def predict (data):
    rf=joblib.load('random_forest_model.pkl')
    return rf.predict(data)


# In[ ]:




