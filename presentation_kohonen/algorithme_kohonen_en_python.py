#!/usr/bin/env python
# coding: utf-8

# ## Application De SOM En Python

# Pour comparer la sécurité des compagnies aériennes, nous avons dû regrouper les neurones de sortie de cartes auto-organisatrice(SOM) à l'aide d'une technique d'apprentissage automatique non supervisée (kmeans) qui regroupe les compagnies aériennes en différents groupes. Ces groupes sont:
# 
# 1. Safe airlines (Compagnies aériennes sûres) <- 0.
# 2. Doubtfully safe airlines (Compagnies aériennes douteuses) <- 1.
# 3. Risky airlines (Compagnies aériennes risquées) <- 2. 

# ### Installation de SOM

# In[ ]:


#!pip install --user -U  SimpSOM


# ### Importation des 

# In[32]:


import pandas as pd
import SimpSOM as sps
from sklearn.cluster import KMeans
import numpy as np


# ### Téléchargement du dataset

# Nous sommes allés télécharger le dataset des compagnies aériennes sur Kaggle sur lien ci-dessous:
# 
# [https://www.kaggle.com/danoozy44/airline-safety](https://www.kaggle.com/danoozy44/airline-safety)

# ### Création de dataset

# In[2]:


df = pd.read_csv("airline-safety.csv")
df.head()


# ### Data préprocessing

# In[3]:


df.describe()


# In[4]:


train = np.array(df.iloc[:, 1:])


# ### Entrainement du modèle Kohonen

# In[33]:


net = sps.somNet(20, 20, train, PBC=True)
net.train(0.01, 200)
net.save("filename_weights")
net.nodes_graph(colnum=0)


# In[7]:


net.diff_graph()


# In[9]:


import matplotlib.pyplot as plt


# In[16]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(prj)
df["clusters"]=kmeans.labels_


# In[70]:


prj=np.array(net.project(train))
plt.scatter(prj.T[0],prj.T[1], c=df.clusters)
plt.show()


# ### Les compagnies aériennes sûrs

# In[39]:


df[df["clusters"]==0].head()


# ### Les compagnies aériennes douteuses

# In[40]:


df[df["clusters"]==1].head()


# ### Les compagnies aériennes à risquées

# In[42]:


df[df["clusters"]==2].head()


# In[53]:


print('{:.2f} % des compagnies sont sûrs'.format(((len(df[df["clusters"]==0])/len(df))*100)))


# In[54]:


print('{:.2f} % des compagnies sont douteuses'.format(((len(df[df["clusters"]==1])/len(df))*100)))


# In[55]:


print('{:.2f} % des compagnies sont risquées'.format(((len(df[df["clusters"]==2])/len(df))*100)))


# ## Résumé

# L'utilisation des SOM suit les étapes suivantes dans la formation du réseau:
# 
# 1. Initialisez les poids des neurones cachés à de petites valeurs aléatoires ou utilisez l'initialisation du poids PCA.
# 2. Alimentez la ligne xi à la couche d'entrée.
# 3. Itérer à travers chaque neurone dans la couche cachée et trouver le BMU et ses unités voisines.
# 4. Appliquez la mise à jour du poids au BMU et à ses neurones voisins.
# 5. Réduire la fonction de voisinage.
# 6. Répétez les étapes 2 à 5 jusqu'à ce que la limite d'itération atteigne ou que le modèle converge.

# ## Réssources
# 1. The Ultimate guide to Self organizing maps (SOM’s) by SuperDataScience Team. link <br>
#    [https://www.superdatascience.com/blogs/the-ultimate-guide-to-self-organizing-maps-soms](https://www.superdatascience.com/blogs/the-ultimate-guide-to-self-organizing-maps-soms)
#    
#    
# 2. Analyzing Climate Patterns with Self-Organizing Maps (SOMs) by Haihan Lan link <br>
#  [https://towardsdatascience.com/analyzing-climate-patterns-with-self-organizing-maps-soms-8d4ef322705b](https://towardsdatascience.com/analyzing-climate-patterns-with-self-organizing-maps-soms-8d4ef322705b)<br/>
#     
#     
# 3. An introduction to self organizing maps by Umut Asan and Secil Ercan link <br>
# [https://www.researchgate.net/publication/263084866_An_Introduction_to_Self-Organizing_Maps](https://www.researchgate.net/publication/263084866_An_Introduction_to_Self-Organizing_Maps)
# 
# 
# 4. Reach our site <br/>[https://www.dalicodes.com/](https://www.dalicodes.com/)
#   

# In[ ]:




