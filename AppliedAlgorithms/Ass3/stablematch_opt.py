#!/usr/bin/python
import math
import random
import time
import matplotlib.pyplot as plt
from collections import deque

class StableMatcher():
  def __init__(self, N):
    self._p = { i: {} for i in range(N) }
    self._p_free = {}
    self._p_size = 0
    self._p_index = {}
    # tracks the preference ranking already assigned for a
    self._a = { i: {} for i in range(N) }
    self._a_free = {}
    self._a_size = 0
    self._a_free = {}
    self._a_index = {}
    # tracks the preference ranking of p for a
    self._a_p = { i: {} for i in range(N) }
    self.threshold = 0.999
    self.N = N
    self.N2 = self.N*self.N
    self.matched = {}

  def generateNextA(self, p):
    p_pref = self._p[p]
    
    # python set operations return a copy, therefore it is O(N) operation.
    # Similarly list <-> set conversions are also O(N), we expect the length of
    # each preference list to be at most log(N), so therefore, we can try using
    # coupon collector algorithm which is just to repeatedly draw numbers until we
    # find a new one.
    if len(p_pref) > self.threshold*self.N:
      if self._prefset == None:
        self._prefset = { i for i in range(N) }
      if p not in self._p_free:
        self._p_free[p] = list(self._prefset - p_pref)
        self._p_index[p] = 0
        for i in range(len(self._p_free[p])):
          j = int(random.uniform(0, i+1))
          tmp = self._p_free[p][i]
          self._p_free[p][i] = self._p_free[p][j]
          self._p_free[p][j] = tmp
      print(f"generateNextA switching to pre-allocation: {len(p_pref)}, {self._p_index[p]}/{len(self._p_free[p])}")
      a = self._p_free[p][self._p_index[p]]
      self._p_index[p] += 1
    else:
      a = None
      while a is None or a in p_pref:
        a = int(random.uniform(0, self.N))

    p_pref.update({a:a})
    self._p_size += 1    
    return a

  def generateRankP(self, a, p):
    a_pref = self._a[a]
    a_p_pref = self._a_p[a]
    if p in a_p_pref:
      return a_p_pref[p]
    if len(a_pref) > self.threshold*self.N:
      if self._prefset == None:
        self._prefset = { i for i in range(N) }
      if a not in self._a_free:
        self._a_free[a] = list(self._prefset - a_pref)
        self._a_index[a] = 0
        for i in range(len(self._a_free[a])):
          j = int(random.uniform(0, i+1))
          tmp = self._a_free[a][i]
          self._a_free[a][i] = self._a_free[a][j]
          self._a_free[a][j] = tmp
      print(f"generateRankP switching to pre-allocation: {len(a_pref)}, {self._a_index[a]}/{len(self._a_free[a])}")
      a = self._a_free[a][self._a_index]
      self._a_index += 1
    else:
      r = None
      while r is None or r in a_pref:
        r = int(random.uniform(0, self.N))

    a_pref.update({r:r})
    a_p_pref.update({p:r})
    self._a_size += 1
    return r

  def analyze(self):
    MRank = 0
    WRank = 0
    for p in self._p:
      MRank += len(self._p[p])
    for a, p in self.matched.items():
      WRank += self._a_p[a][p]
    return MRank/self.N, WRank/self.N  
  
  def match(self, log=False):
    """
    Args:
      log - True to print trace of matchings
    """
    # every proposer start free with their top preference
    p_free = deque( [i for i in range(self.N)] )

    while len(p_free) > 0:
      # the order of proposers do not matter, we pick first from a random permutation
      p = p_free.popleft()
      st = time.time()
      w = self.generateNextA(p)
      wGenTime = time.time() - st
      if w not in self.matched:
        self.generateRankP(w, p)
        self.matched[w] = p
      else:
        p_prime = self.matched[w]
        st = time.time()
        p_index = self.generateRankP(w, p)
        p_prime_index = self.generateRankP(w, p_prime)
        mGenTime = time.time() - st
        if p_index < p_prime_index:
          self.matched[w] = p
          p_free.append(p_prime)
        else:
          p_free.append(p)
      if log:
        if len(p_free) % 50000 == 0 and len(p_free) != 0:
          print("wGenTime:", wGenTime, "mGenTime:", mGenTime)
          print("p_free:", len(p_free), "p_size:", self._p_size,
				"a_size:", self._a_size, "total:", self.N2 )
        elif len(p_free) < 50000 and len(p_free) % 1000 == 0 and len(p_free) != 0:
          print("wGenTime:", wGenTime, "mGenTime:", mGenTime)
          print("p_free:", len(p_free), "p_size:", self._p_size,
				"a_size:", self._a_size, "total:", self.N2 )
        elif len(p_free) < 1000 and len(p_free) % 100 == 0 and len(p_free) != 0:
          print("wGenTime:", wGenTime, "mGenTime:", mGenTime)
          print("p_free:", len(p_free), "p_size:", self._p_size,
				"a_size:", self._a_size, "total:", self.N2 )
        elif len(p_free) < 100:
          print("wGenTime:", wGenTime, "mGenTime:", mGenTime)
          print("p_free:", len(p_free), "p_size:", self._p_size,
				"a_size:", self._a_size, "total:", self.N2 )

if __name__ == "__main__":
  random.seed()
  #N = [500000, 1000000, 2000000, 4000000, 8000000]
  N = [3000000, 4000000]
  repetition = 1

  runtimes = []
  mGoodAvgs = []
  mTheoretical = []
  wGoodAvgs = []
  wTheoretical = []
  for n in N:
    print(f"Size {n}")
    mGoodAvg = 0
    wGoodAvg = 0
    runtime = 0
    for j in range(repetition):
      print(f"repetition {j}")
      print("matching...")
      sm = StableMatcher(n)
      st = time.time()
      sm.match(log=True)
      runtime += time.time() - st
      mGood, wGood = sm.analyze()
      mGoodAvg += mGood
      wGoodAvg += wGood
      print("runtime...", runtime)
    runtimeAvg = runtime / repetition
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
  plt.ylabel("Rank/N")
  plt.xlabel("N")
  plt.legend()

  plt.figure()
  plt.plot(N, runtimes, "-p", label="runtime")
  plt.ylabel("time (s)")
  plt.xlabel("N")
  plt.legend()

  plt.show() 
