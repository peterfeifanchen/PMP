#!/usr/bin/python

import random
import statistics
import math

class QSelect():
  def __init__(self, medianN=1):
    self.medianN = medianN
    # Count of comparisons needed to sort into sets larger, smaller and equal to the pivot
    self.compSet = 0
    # Count of comparisons needed to decide branching in the recursion
    self.compPivot = 0
 
  # Gets the k-th largest item.
  # NOTE: median is k=N/2
  # NOTE: k-th largest item is counted from the right.
  def __call__(self, seq, k):
    # Randomly choose N elements from sequence
    medians = []
    for _ in range(self.medianN):
      medians.append(seq[int(random.uniform(0, len(seq)))])
    # We are only using odd numbers for medianN, may run into problems if it was even
    # and statistics.median returns the average of the middle two numbers. It might also
    # run into problems when k is not N/2, might not be able to use statistics.median.
    x = statistics.median(medians)
 	
    # Create separate sets
    s1 = [] # seq[i] < x
    s2 = [] # seq[i] == x
    s3 = [] # seq[i] > x
    for value in seq:
      if value < x:
        s1.append(value)
      elif value == x:
        s3.append(value)
      else:
        s2.append(value)
    self.compSet += len(seq)
    if len(s2) >= k:
      self.compPivot += 1
      return self(s2, k)
    elif len(s2) + len(s3) >= k:
      return x
    else:  
      self.compPivot += 1
      return self(s1, k - len(s2) - len(s3)) 

  def reset(self):
    self.compSet = 0
    self.compPivot = 0

if __name__ == "__main__":
  random.Random(None)

  # Generate inputs
  medianOfMedians = 5
  N = 100
  sequence = [i for i in range(N)]

  q = QSelect(medianOfMedians)
  median_qselect = q(sequence, int(math.ceil(N/2)))
  median_expected = sequence[int(N/2)]
  print( "median (qselect)", median_qselect )
  print( "median (expected)", median_expected )
  assert median_qselect == median_expected
  print( "qselect compSet:", q.compSet/N, "qselect compPivot", q.compPivot ) 

  # Get average time complexity with 100 trials 
  T = 100 # number of trials
  medianOfMedians = 5 
  N = 10000000
  sequence = [i for i in range(N)]

  compSets = []
  compPivots = []
  
  q = QSelect(medianOfMedians)
  for _ in range(100):
    q.reset()
    q(sequence, int(math.ceil(N/2)))
    compSets.append(q.compSet/N)
    compPivots.append(q.compPivot)

  print( "mean compSets:", statistics.mean(compSets))
  print( "mean compPivots:", statistics.mean(compPivots))   
