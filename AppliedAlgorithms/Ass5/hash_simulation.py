#!/usr/bin/python
import math
import random
import matplotlib.pyplot as plt
import numpy as np

class BloomFilter():
  def __init__(self, m=10000, n=1000, iterations=1000000, method="random"):
    self._k = round( m/n * math.log(2) ) #optimal k given m and n
    self._B = n/m
    self._slots = [0] * m
    self._m = m
    self._iterations = iterations
    self._values = n*n # n^2 different values
    self._empty = m
    self._n = n
    self._deletion_method = method #random or fifo
    self._fifo = np.random.uniform(0, self._values, n)
    self._head = 0
    self._locations = {} # simulate hashing the same item to the same location 
 
  def _add(self, i):
    if i not in self._locations:
      self._locations[i] = [ int(random.uniform(0, self._m)) for _ in range(self._k) ]
    ilocs = self._locations[i]

    for iloc in ilocs:
      if self._slots[iloc] == 0:
        self._empty -= 1
      self._slots[iloc] += 1

  def _delete(self, i):
    if i not in self._locations:
      self._locations[i] = [ int(random.uniform(0, self._m)) for _ in range(self._k) ]
    ilocs = self._locations[i]

    for iloc in ilocs:
      if self._slots[iloc] > 0:
        self._slots[iloc] -= 1
        if self._slots[iloc] == 0:
          self._empty += 1

  def run(self):
    # We fill the BloomFilter to half full, then iterate self._n number of
    # times with random deletion or fifo deletion followed by another random
    # add.
    
    # 1) pre-warm the bloom filter with self._fifo
    for i in self._fifo:
      self._add(i)    

    empty_slots = [ self._empty ]

    # 2) run random/fifo delete and add for iterations
    for n in range(self._iterations):
      item_add = random.uniform(0, self._values)
      if self._deletion_method == "random":
        item_delete = random.uniform(0, self._values)
      elif self._deletion_method == "fifo":
        item_delete = self._fifo[self._head]
        self._fifo[self._head] = item_add
        self._head = (self._head + 1) % self._n
     
      self._delete(item_delete)
      self._add(item_add)
      empty_slots.append(self._empty) 
         

    return self._slots, empty_slots

if __name__ == "__main__":
  np.random.seed()
  random.seed()

  m = 10000
  n = 2000
  iterations = 1000000
  method = "random"
  bf = BloomFilter(m=m, n=n, iterations=iterations, method=method)
  slotcount, emptycount = bf.run()

  plt.figure()
  plt.hist(slotcount, color='blue', edgecolor='black', bins=len(slotcount))
  plt.ylabel( "number of slots" )
  plt.xlabel( "slot count" )
  plt.title( f"Counting Bloom Filter n={n} m={m} delete={method} B={bf._B:.2f}" )

  plt.figure()
  plt.plot([ i for i in range(len(emptycount)) ], emptycount, '-', label="empty slot count")
  plt.ylabel("empty slot count")
  plt.xlabel("iteration")
  plt.title( f"Counting Bloom Filter n={n} m={m} delete={method} B={bf._B:.2f}" )
  plt.legend()
  plt.show() 
