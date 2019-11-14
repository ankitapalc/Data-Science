# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:38:55 2019

@author: palan
"""

# Importing Libraries and cookbooks
#from recsys import * ## recommender system cookbook
#from generic_sys import * ## pre-processing code
#from IPython.display import HTML ## Setting display options for Ipython Notebook
import pandas as pd,numpy as np

anime = pd.read_csv('anime.csv')
ratings = pd.read_csv('rating.csv').iloc[:100000]

ids = np.unique(ratings.anime_id)

anime.head()

anime_dict = {}

for i,el in enumerate(anime.anime_id.values):
  if(el in ids):
    #print(el)
    anime_dict[el] = anime.name.values[i]
    #break

print(anime_dict)


# Creating interaction matrix using rating data
interactions = create_interaction_matrix(df = ratings,
                                         user_col = 'user_id',
                                         item_col = 'anime_id',
                                         rating_col = 'rating')
interactions.shape

mf_model = runMF(interactions = interactions,
                 n_components = 30,
                 loss = 'warp',
                 epoch = 30,
                 n_jobs = 4)

## Creating item-item distance matrix
item_item_dist = create_item_emdedding_distance_matrix(model = mf_model,interactions = interactions)
## Checking item embedding distance matrix
#item_item_dist.head()


## Calling 10 recommended items for item id
rec_list = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,
                                    item_id = 1535,
                                    item_dict = anime_dict,
                                    n_items = 10)

l = anime.name[ratings.anime_id.unique()]

q = input('anime key: ')

for i in anime_dict:
  if(q in anime_dict[i].lower()):
    print(i,anime_dict[i])