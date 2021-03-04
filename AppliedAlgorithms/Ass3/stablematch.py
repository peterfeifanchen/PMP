#!/usr/bin/python
import math
import random
import time
import matplotlib.pyplot as plt
from collections import deque

def generate(N: int):
  arr = [i for i in range(N)]

  for i in range(N):
    j = int(random.uniform(0, i+1))
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp

  return arr
 
class StableMatcher():
  """
  p_matrix: proposer matrix
  a_matrix: accepter matrix
  """
  def __init__(self, p_matrix, a_matrix):
    self._p = p_matrix
    self._a = a_matrix
    self.N = len(p_matrix)

  def analyze(self, results: dict, p_rank: list, a_matrix_invert=False):
    MRank = 0
    WRank = 0

    for i in p_rank:
      MRank += i

    for w, m in results.items():
      if a_matrix_invert:
        WRank += self._a[w].index(m)
      else:
        WRank += self._a[w][m]
    
    return MRank/self.N, WRank/self.N  
  
  def next_p(self, start, l):
    for i in range(start, len(l)):
      if l[i] == 1:
        return i
    for i in range(0, start):
      if l[i] == 1:
        return i

  def match(self, a_matrix_invert=False, log=False):
    """
    Args:
      a_matrix_invert - columns of a_matrix are different proposers or rank
        false - proposer id
        true - rank
      log - True to print trace of matchings
    """
    if log:
      print("========================================")
      print("Input (Proposer):")
      print(self._p)
      print("Input (Accepter):")
      print(self._a)
      print("Trace")

    # every proposer start free with their top preference
    p_free = deque( [ i for i in range(self.N) ] )
    p_next = [ 0 for _ in range(self.N) ]
    matched = {}

    p = 0
    st = time.time()
    while len(p_free) > 0:
      # the order of proposers do not matter, we pick first from a random permutation
      p = p_free.popleft()
      w = self._p[p][p_next[p]]

      if w not in matched:
        matched[w] = p
        if log:
          print(f"{p} proposed to {w} [{w}, -1] Accepted")
      else:
        p_prime = matched[w]
        if a_matrix_invert:
          p_index = self._a[w].index(p)
          p_prime_index = self._a[w].index(p_prime)
        else:
          p_index = self._a[w][p]
          p_prime_index = self._a[w][p_prime]
        if p_index < p_prime_index:
          matched[w] = p
          p_free.append(p_prime)
          if log:
            print(f"{p} proposed to {w} [{w}, {p_prime}] Accepted")
        else:
          p_free.append(p)
          if log:
            print(f"{p} proposed to {w} [{w}, {p_prime}] Rejected")
      p_next[p] += 1

    if log:
      print("Output:")
      print(matched) 

    return matched, p_next, time.time() - st     

if __name__ == "__main__":

#  # Sanity Check
#  m_pref = [[2,0,1],[2,0,1],[2,0,1]]
#  w_pref = [[2,0,1],[0,2,1],[2,0,1]]
#  sm = StableMatcher( m_pref, w_pref )
#  sm.match(log=True)
#  
#  m_pref = [[3,1,0,2],[3,0,1,2],[3,1,2,0],[0,3,2,1]]
#  w_pref = [[3,0,2,1],[3,1,2,0],[0,1,3,2],[3,0,2,1]]
#  sm = StableMatcher( m_pref, w_pref )
#  sm.match(log=True)
#
#  m_pref = [[1,3,0,2],[3,0,2,1],[2,3,1,0],[2,0,3,1]]
#  w_pref = [[1,2,0,3],[0,3,2,1],[1,2,3,0],[0,1,2,3]]
#  sm = StableMatcher( m_pref, w_pref )
#  sm.match(log=True)
#
#  m_pref = [[4,2,0,1,3],[0,1,4,2,3],[4,0,1,3,2],[1,4,3,0,2],[0,1,3,4,2]]
#  w_pref = [[2,3,4,0,1],[2,4,0,3,1],[1,4,2,3,0],[3,0,1,4,2],[0,3,1,2,4]]
#  sm = StableMatcher( m_pref, w_pref )
#  sm.match(log=True)

  # Question 4
  m_pref = [[2,1,3,0],[0,1,3,2],[0,1,2,3],[0,1,2,3]]
  w_pref = [[0,2,1,3],[2,0,3,1],[3,2,1,0],[2,3,1,0]]

  sm = StableMatcher( m_pref, w_pref )
  sm.match(a_matrix_invert=True, log=True)

  random.seed()
  N = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
  #N = [100, 200, 400, 800]
  repetition = 3

  runtimes = []
  mGoodAvgs = []
  mTheoretical = []
  wGoodAvgs = []
  wTheoretical = []
  for n in N:
    print(f"Size {n}")
    mGoodAvg = 0
    wGoodAvg = 0
    runtimeAvg = 0
    for j in range(repetition):
      print(f"repetition {j}")
      m_pref = [ generate(n) for _ in range(n) ]
      w_pref = [ generate(n) for _ in range(n) ]
      print("matching...")
      sm = StableMatcher( m_pref, w_pref )
      m, p_next, runtime = sm.match(a_matrix_invert=False, log=False)
      mGood, wGood = sm.analyze(results=m, p_rank=p_next, a_matrix_invert=False)
      mGoodAvg += mGood
      wGoodAvg += wGood
      runtimeAvg += runtime
    runtimeAvg = runtimeAvg / repetition
    mGoodAvg = mGoodAvg / repetition
    wGoodAvg = wGoodAvg / repetition
    runtimes.append(runtimeAvg)
    mGoodAvgs.append(mGoodAvg)
    wGoodAvgs.append(wGoodAvg)
    mTheoretical.append(math.log(n))
    wTheoretical.append(n/math.log(n))

  plt.figure()
  plt.plot(N, mGoodAvgs, "p", label="men (experimental)")
  plt.plot(N, wGoodAvgs, "p", label="women (experimental)")
  plt.plot(N, mTheoretical, "-", label="men (theoretical)")
  plt.plot(N, wTheoretical, "-", label="women (theoretical)")
  plt.ylabel("Rank/n")
  plt.xlabel("N")
  plt.legend()

  plt.figure()
  plt.plot(N, runtimes, "-p", label="runtime")
  plt.ylabel("time (s)")
  plt.xlabel("N")
  plt.legend()

  plt.show() 
