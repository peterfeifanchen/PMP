#!/usr/bin/python
import math
import random
import matplotlib.pyplot as plt

class HashSimulator():
  def __init__(self, N, hashes):
    self._N = N
    self._hashes = hashes

  def simulate(self, runs=1):
    maxSlotSizeAvg = 0
    for _ in range(runs):
      maxSlotSize = 0
      hash_table = [ 0 for i in range(self._N) ]
      for _ in range(self._N):
        indices = [ int(random.uniform(0, self._N)) for i in range(self._hashes) ]
        index_sizes = [ (i, hash_table[i]) for i in indices ]
        index = min( index_sizes, key=lambda x: x[1] )[0]
        hash_table[index] += 1
        if hash_table[index] > maxSlotSize:
          maxSlotSize = hash_table[index]
      maxSlotSizeAvg += maxSlotSize
    return maxSlotSizeAvg / runs
 
if __name__ == "__main__":
#  random.seed()
#  N = 1000000
#  h = 3
#  runs = 10
#  hs = HashSimulator(N, h)
#  res = hs.simulate(runs)
#  print("simulated:", res)
#  if h > 1:
#    print("theoretical:", math.log2(math.log2(N)))
#  else:
#    print("theoretical:", math.log2(N)/math.log2(math.log2(N)))

  N = [1000, 10000, 100000, 1000000]
  hashes = [1, 2, 3]
  runs = 25
  experimental_results = []
  theoretical_results = []
  for h in hashes:
    experimental_results.append([])
    theoretical_results.append([])
    for n in N:
      if h == 1:
        theoretical_results[-1].append(math.log2(n)/math.log2(math.log2(n)))
      else:
        theoretical_results[-1].append(math.log2(math.log2(n)))
      hs = HashSimulator(n,h)
      res = hs.simulate(runs)
      print(f"hashes: {h} size: {n} avg: {res}")
      experimental_results[-1].append(res) 

  print("all experimental results:", experimental_results)
  print("all theoretical results:", theoretical_results)
  plt.figure()
  plt.plot(N, theoretical_results[0], "-", label="1-choice hash theoretical")
  plt.plot(N, theoretical_results[1], "-", label="2-choice hash theoretical")
  plt.plot(N, theoretical_results[2], "-", label="3-choice hash theoretical")
  plt.plot(N, experimental_results[0], "-p", label="1-choice hash experimental")
  plt.plot(N, experimental_results[1], "-p", label="2-choice hash experimental")
  plt.plot(N, experimental_results[2], "-p", label="3-choice hash experimental")
  plt.ylabel("max slot size")
  plt.xlabel("N")
  plt.legend()
  plt.show() 
  
