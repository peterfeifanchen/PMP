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
  def __init__(self):
    # Parse the input files into a sparse matrix
    data_file = "data50.csv"
    label_file = "label.csv"
    groups_file = "groups.csv"

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

  def plotAvg(self, measure="jaccard"):
    """
    Plot average similarity between items in group A and B
    measure:
      jaccard: jaccard similarity
      cosine : cosine similarity
      l2     : L2 similarity
    """
    categories = list(self.data.keys())
    similarity = np.zeros((len(categories), len(categories)))
    for i in range(len(categories)):
      for j in range(len(categories)):
        articlesA = self.data[categories[i]]
        articlesB = self.data[categories[j]]    
        avg = 0
        for articleA in articlesA:
          for articleB in articlesB:
            avg += self._norm( articlesA[articleA], articlesB[articleB], measure )
        avg = avg / (len(articlesA) * len(articlesB))      
    
        similarity[i][j] = avg

    makeHeatMap(similarity, categories, 'RdBu', 'similarity_avg' + measure + ".png")
 
  def plotMostSimilar(self):
    """
    Plot count of articles form B that are most similar to any article in A
    based on Jaccard similarity
    """
    categories = list(self.data.keys())
    similarity = np.zeros((len(categories), len(categories)))
    most_like_count = { c: {} for c in categories }
    
    for i in range(len(categories)):
      articlesA = self.data[categories[i]] 
      for articleA, featuresA in articlesA.items():
        most_like_count[ categories[i] ][articleA] = (0, 'unk')
        for j in range(len(categories)):
          if i == j:
            continue
          articlesB = self.data[categories[j]]
          for articleB, featuresB in articlesB.items():
            s = self._norm( featuresA, featuresB, "cosine" )
            if s > most_like_count[categories[i]][articleA][0]:
              most_like_count[categories[i]][articleA] = (s, j) 

    for i in range(len(categories)):
      for article, most_similar_article in most_like_count[categories[i]].items():
        most_similar_article_category = most_similar_article[1]
        if most_similar_article_category != 'unk':
          similarity[i][most_similar_article_category] += 1 

    #makeHeatMap(similarity, categories, 'RdBu', 'similarity_mostlike_jaccard.png')
    makeHeatMap(similarity, categories, 'RdBu', 'similarity_mostlike_cosine_exact.png')

  def _norm(self, A, B, measure):
    """
    Calculate the norm given measure 
    """
    if measure == "jaccard":
      return self._jaccard(A, B)
    elif measure == "cosine":
      return self._cosine(A,B)
    elif measure == "l2":
      return self._l2(A,B)
    else:
      sys.exit("invalid measure")

  def _jaccard(self, A, B):
    minCount = {}
    maxCount = {}
    for word, count in A.items():
      if count > B[word]:
        minCount[word] = B[word]
        maxCount[word] = count
      else:
        minCount[word] = count
        maxCount[word] = B[count]

    for word, count in B.items():
      if count > A[word]:
        minCount[word] = A[word]
        maxCount[word] = count
      else:
        minCount[word] = count
        maxCount[word] = A[word]

    sumMin = 0
    sumMax = 0

    for _, count in minCount.items():
      sumMin += count

    for _, count in maxCount.items():
      sumMax += count

    return sumMin/sumMax

  def _cosine(self, A, B):
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

  def _l2(self, A, B):
    checked = set()
    norm_sum = 0
    for word, count in A.items():
      Bcount = B[word]
      norm_sum += ( count - Bcount ) * ( count - Bcount )

    for word, count in B.items():
      if word not in checked:
        norm_sum += count * count

    return -math.sqrt(norm_sum) 

if __name__ == "__main__":
  st = time.time()
  s = Similarity()
  #s.plotAvg()
  #s.plotAvg('l2')
  #s.plotAvg('cosine')
  s.plotMostSimilar()
  print("runtime", time.time() - st)
