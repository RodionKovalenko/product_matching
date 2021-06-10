#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# # CSV Import

# In[2]:


comp = pd.read_csv("data/Productnames/competition_products.csv")


# In[3]:


# Überschrift 4-6 nicht erkennbar und nicht relevant
basis = pd.read_csv("data/Productnames/store_products.csv",names=["id","product", "Store", "NAN", "NAN2","NAN3"])
basis = basis[['id','product','Store']]


# In[4]:


basis


# In[5]:


comp


# # Data Cleaning / EDA

# ### basis Table

# In[6]:


basis.info()




# In[8]:


# remove product = nan
product_not_null = pd.notnull(basis["product"])
basis=basis[product_not_null]



# ### comp Table

# In[10]:


comp.info()


# In[11]:




# In[12]:


enabled_not_null = pd.notnull(comp['enabled'])
comp=comp[enabled_not_null]
#comp['enabled'].dropna()


#  # Join of the two Dataframes

# In[13]:


Merge = pd.merge(basis,comp,left_on='id',right_on='product_id')
Merge


# ## Formatting 

# In[14]:


Merge = Merge.drop(['id_x', 'id_y','description'], axis=1)
dict = {'product': 'productname_1',
        'Store': 'seller_1',
        'name': 'productname_2',
        'seller': 'seller_2',
        'enabled': 'match'}
Merge.rename(columns=dict,inplace=True)


# In[15]:


Merge = Merge[['product_id', 'productname_1', 'seller_1', 'productname_2', 'seller_2','match']]


# In[16]:


Merge.to_csv("product_matching.csv", index=False)

