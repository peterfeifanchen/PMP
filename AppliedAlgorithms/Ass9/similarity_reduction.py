#!/usr/bin/python

import time
import random
import csv
import sys
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import Counter
from heatmap import makeHeatMap

class Similarity():
  def __init__(self, d, reduction="jl"):
    # Parse the input files into a sparse matrix
    data_file = "data50.csv"
    label_file = "label.csv"
    groups_file = "groups.csv"

    n = 0
	# Data Structure
    # Group:
    #   Article: Counter( word: frequency )
    self.data = {}

    group = []
    with open(groups_file) as csvfile:
      groups_reader = csv.reader(csvfile, delimiter = '\n')
      for row in groups_reader:
        self.data[row[0]] = {}
        group.append(row[0])

    label = []
    with open(label_file) as csvfile:
      label_reader = csv.reader(csvfile, delimiter = '\n')
      for row in label_reader:
        label.append(int(row[0]))
    
    with open(data_file) as csvfile:
      data_reader = csv.reader(csvfile, delimiter = ',')
      for row in data_reader:
        row_id = int(row[0])
        group_id = label[row_id - 1]
        group_name = group[group_id - 1]
        if row_id not in self.data[group_name]:
          self.data[group_name][row_id] = Counter()
        self.data[group_name][row_id][int(row[1])]= int(row[2])
        if n < int(row[1]):
          n = int(row[1])
    
	# Data Structure
    # Group:
    #   Article: d-vector fingerprint
    if reduction == "jl":
      # use random normal hash
      M = np.random.normal(0,1, (n, d))
    elif reduction == "uniform":
      M = np.random.uniform(-1,1, (n,d))
      M[M>0] = 1
      M[M<=0] = -1
    self.d = d
    self.reduction = reduction
    self.data_reduced = {}
    for group, articles in self.data.items():
      if group not in self.data_reduced:
        self.data_reduced[group] = {}
      for article, c in articles.items():
        v = np.zeros(n)
        for index, val in c.items():
          v[index-1] = val
        h = v.dot(M)
        h[h>0] = 1
        h[h<0] = -1
        self.data_reduced[group][article] = h

  def plotMostSimilar(self):
    """
    Plot count of articles form B that are most similar to any article in A
    based on cosine similarity
    """
    categories = list(self.data_reduced.keys())
    similarity = np.zeros((len(categories), len(categories)))
    most_like_count = { c: {} for c in categories }
    
    for i in range(len(categories)):
      articlesA = self.data_reduced[categories[i]] 
      for articleA, featuresA in articlesA.items():
        most_like_count[ categories[i] ][articleA] = (0, 'unk')
        for j in range(len(categories)):
          if i == j:
            continue
          articlesB = self.data_reduced[categories[j]]
          for articleB, featuresB in articlesB.items():
            s = self._cosine( featuresA, featuresB )
            if s > most_like_count[categories[i]][articleA][0]:
              most_like_count[categories[i]][articleA] = (s, j) 

    for i in range(len(categories)):
      for article, most_similar_article in most_like_count[categories[i]].items():
        most_similar_article_category = most_similar_article[1]
        if most_similar_article_category != 'unk':
          similarity[i][most_similar_article_category] += 1 

    makeHeatMap(similarity, categories, 'RdBu', f'similarity_mostlike_cosine_jl_{self.d}.png')

  def scatterplot(self, article=3, group="alt.atheism"):
    ref_reduced_features = self.data_reduced[group][article]
    ref_features = self.data[group][article]

    x = []
    y = []
    for group, articles in self.data.items():
      for article, features in articles.items():
        s = self._cosine_exact( ref_features, features )
        x.append(s)
        s = self._cosine( ref_reduced_features, self.data_reduced[group][article] )
        y.append(s)
   
    plt.figure()
    plt.plot(x, y, 'p')
    plt.plot(x, x, '-')
    plt.ylabel(f"cosine similarity d={self.d}")
    plt.xlabel("cosine similarity exact")
    plt.savefig(f"scatter_{self.reduction}_{self.d}.png", format='png')

  def _cosine_exact(self, A, B):
    X2 = 0
    Y2 = 0
    dotXY = 0

    checked = set()
    for word, count in A.items():
      Bcount = B[word]
      dotXY += Bcount * count
      X2 += count * count
      Y2 += Bcount * Bcount
      checked.add(word)

    for word, count in B.items():
      if word not in checked:
        Y2 += count * count

    return dotXY / ( math.sqrt(X2) * math.sqrt(Y2) )
  
  def _cosine(self, A, B):
    if np.linalg.norm(A) * np.linalg.norm(B) == 0:
      return 0.0
    return A.dot(B) / (np.linalg.norm(A) * np.linalg.norm(B))


if __name__ == "__main__":
  st = time.time()
  #s = Similarity(d=10)
  s = Similarity(d=100, reduction="uniform")
  #s.plotMostSimilar()
  s.scatterplot()
  print("runtime", time.time() - st)
