#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df_X_train_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_train_transformed.csv')
df_X_train_transformed


# In[23]:


df_X_train_transformed.describe().T


# In[6]:


df_Y_train_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_train_transformed.csv')
df_Y_train_transformed


# In[2]:


df_X_test_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_test_transformed.csv')
df_X_test_transformed


# In[24]:


df_X_test_transformed.describe().T


# In[7]:


df_Y_test_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_test_transformed.csv')
df_Y_test_transformed


# In[5]:


df_X_val_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_val_transformed.csv')
df_X_val_transformed


# In[25]:


df_X_val_transformed.describe().T


# In[8]:


df_Y_val_transformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_val_transformed.csv')
df_Y_val_transformed


# In[ ]:





# In[ ]:





# In[9]:


df_X_train_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_train_untransformed.csv')
df_X_train_untransformed


# In[36]:


from IPython.display import display


# In[44]:


with pd.option_context('display.max_rows', df_X_train_untransformed.shape[1]):
    display(df_X_train_untransformed.describe().T)


# In[10]:


df_Y_train_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_train_untransformed.csv')
df_Y_train_untransformed


# In[11]:


df_X_test_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_test_untransformed.csv')
df_X_test_untransformed


# In[46]:


with pd.option_context('display.max_rows', df_X_test_untransformed.shape[1]):
    display(df_X_test_untransformed.describe().T)


# In[12]:


df_Y_test_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_test_untransformed.csv')
df_Y_test_untransformed


# In[13]:


df_X_val_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/X_val_untransformed.csv')
df_X_val_untransformed


# In[48]:


with pd.option_context('display.max_rows', df_X_val_untransformed.shape[1]):
    display(df_X_val_untransformed.describe().T)


# In[14]:


df_Y_val_untransformed = pd.read_csv('/Users/shreyandattachakraborty/Documents/College Application/Application for Masters/OSU/Study Materials/2nd Semester/MSIS 5223/Assignments/project-deliverable-2-cia/data/predictive_modeling_data/Y_val_untransformed.csv')
df_Y_val_untransformed


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




