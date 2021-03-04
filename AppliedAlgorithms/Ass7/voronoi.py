#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import math

def distance_norm_l1( pA, pB ):
  return abs(pA[0]-pB[0]) + abs(pA[1]-pB[1])

def distance_norm_inf( pA, pB ):
  return max( abs(pA[0]-pB[0]), abs(pA[1]-pB[1]) )

def distance_norm_l2( pA, pB ):
  d_2 = (pA[0]-pB[0])*(pA[0]-pB[0]) + (pA[1]-pB[1])*(pA[1]-pB[1])
  return math.sqrt(d_2)

def predict( points, x, y ):
  cm = np.zeros(points.shape[0])
  for j in range(len(points)):
    minDistance = distance_norm_l1((x[0],y[0]), points[j])
    minLabel = 0
    for i in range(len(x)):
      d = distance_norm_l1((x[i], y[i]), points[j])
      if d < minDistance:
        minLabel = i
        minDistance = d
    cm[j] = minLabel
  return cm 

if __name__ == "__main__":
  points_x = [0, 1, 1, 3, 4]
  points_y = [2, 0, 4, 3, 2]

  xx, yy = np.meshgrid(np.arange(-1, 5, 0.1), np.arange(-1, 5, 0.1))
  Z = predict(np.c_[xx.ravel(), yy.ravel()], points_x, points_y).reshape(xx.shape)
  plt.figure()
  plt.xlim([-1,5])
  plt.ylim([-1,5])
  plt.contour(xx, yy, Z, alpha=0.4)
  plt.plot(points_x, points_y, 'o', color='tab:red')
  plt.show()
