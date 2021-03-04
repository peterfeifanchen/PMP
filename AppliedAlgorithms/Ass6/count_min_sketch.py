#!/usr/bin/python
import math
import time
import random
import statistics
import matplotlib.pyplot as plt
from collections import Counter

# Carter-Wegman universal hash functions
# 
class Hasher():
  """
  Carter-Wegman universal hash functions. For any x, it hashes x to ax+b mod p
  where p is a prime, and a and b are integers in range [1...p-1] randomly chosen
  when hash function is initiated.
  """
  def __init__(self, buckets: int):
    self._p = 2147483629  # prime less than 2^31
    self._p1 = 4294967291 # prime less than 2^32
    self._p2 = 65521      # prime less than 2^16  

    self._a = int(random.uniform(1, self._p))
    self._b = int(random.uniform(1, self._p))
    self._buckets = buckets

  def __call__(self, s: str):
    """
    strings are converted to integers by treating the characters as coefficients of
    a polynomial, which is then evaluated at a fixed value. This arithmetic is
    again done mod a different prime p'. 
    """
    val = 0
    x = 1
    for i in range(len(s)):
      c = ord(s[i])
      val = (val + c * x) % self._p1
      x = (x * self._p2) % self._p1 

    result = ( val * self._a + self._b ) % self._p
    return result % self._buckets 
   
class FakeHasher():
  """
  Simulated hash that randomly spreads results for each object and remembers where
  it last stored them.
  """
  def __init__(self, buckets: int):
    self._index = {} # remember where something was assigned
    self._buckets = buckets

  def __call__(self, s: str):
    if s not in self._index:
      self._index[s] = int(random.uniform(0, self._buckets))
    return self._index[s]
    
 
class CountMinSketch():
  def __init__(self, k: int, buckets: int , hashes: int):
    self.b = buckets
    self.epsilon = 1 / buckets
    self.h = [ FakeHasher(buckets) for _ in range(hashes) ]
    self.k = k # identify objects occuring n/k times
    self.count_cms = [[0 for _ in range(buckets)] for _ in range(hashes)] # Count-Min-Sketch table
    self.count_approx = Counter()
    self.count_precise = Counter() # Precise count table
    self.n = 0 # total number of datapoints seen in the stream

  def process(self, data: str):
    """
    process data to add to Count-Min-Sketch and keep an actual count
    for comparison
    """
    # add to precise count 
    self.count_precise[data] += 1

    # add to min-sketch count
    for i in range(len(self.h)):
      pos = self.h[i](data)
      self.count_cms[i][pos] += 1 

    self.n += 1  

  def cms_count(self, data: str):
    """
    return the precise and cms count of the data item
    """
    minCount = self.n
    for i in range(len(self.h)):
      pos = self.h[i](data)
      if self.count_cms[i][pos] < minCount:
        minCount = self.count_cms[i][pos]
    
    return minCount    

  def analyze(self):
    """
    Analyze error rate of all items that appear > n/k times, we look
    at the overcount for these items (bucket them into histograms and
    average overcount).
    NOTE: this overcount average is not weighted by occurence of each
    item.

    Given theoretical analysis, we expect the count to exceed 2*n/b less
    than 50% of time, with expected overcount of n/b.  

    Precision: true positives / ( true positives + false positives ) where
    positive is a number cms found to have occured n/k times and actually
    did occur n/k times

    Recall: true positives / ( true positives + false negatives )
    """
    overcount = []
    cnt_dist = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    overcnt_threshold = 0
     
    threshold = self.n / self.k
    pct_50_threshold = 2 * self.n / self.b
    for data, precise_count in self.count_precise.items():
      cnt_dist.append(precise_count)
      cms_count = self.cms_count(data)
      self.count_approx[data] = cms_count
      overcount.append( cms_count - precise_count )
      #if cms_count - precise_count > pct_50_threshold and precise_count > threshold - pct_50_threshold:
      if cms_count - precise_count > pct_50_threshold:
        overcnt_threshold += 1

      if precise_count >= threshold and cms_count >= threshold:
        true_positives += 1
      elif precise_count >= threshold and cms_count < threshold:
        false_negatives += 1
      elif precise_count < threshold - pct_50_threshold and cms_count > threshold:
        # we define false positives to at n/k - epsilon*n
        false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    pct_50_prob_theoretical = 0.5 ** len(self.h) #upper bound
    pct_50_prob_experimental = overcnt_threshold / len(self.count_precise)

    overcount_expected = self.n / self.b
    overcount_experimental = statistics.mean(overcount)

    #overcnt_avg_most_common = 0 
    #for item, count in self.count_precise.most_common(25):
    #  approx = self.count_approx[item]
    #  overcnt_avg_most_common += approx - count
    #  print(item, ":", count, approx, approx - count)
    #print("Total Over Count:", overcnt_avg_most_common)
    #print("Total Items:", len(self.count_precise)) 
    return {
      "cnt_dist": cnt_dist,
      "overcnt_dist": overcount,
      "precision": precision,
      "recall": recall,
      "pct_50_threshold": pct_50_threshold,
      "pct_50_prob_theoretical": pct_50_prob_theoretical,
      "pct_50_prob_experimental": pct_50_prob_experimental,
      "overcnt_theoretical": overcount_expected,
      "overcnt_experimental": overcount_experimental,
    }

class Graph():
  def __init__(self, x, xlabel, title):
    self.precision = []
    self.recall = []
    self.pct_50_threshold = []
    self.pct_50_prob_theoretical = []
    self.pct_50_prob_experimental = []
    self.overcnt_theoretical = []
    self.overcnt_experimental = []
    self.x = x
    self.xlabel = xlabel
    self.title = title

  def append_result(self, result):
    self.precision.append(result["precision"])
    self.recall.append(result["recall"])
    self.pct_50_threshold.append(result["pct_50_threshold"])
    self.pct_50_prob_theoretical.append(result["pct_50_prob_theoretical"])
    self.pct_50_prob_experimental.append(result["pct_50_prob_experimental"])
    self.overcnt_theoretical.append(result["overcnt_theoretical"])
    self.overcnt_experimental.append(result["overcnt_experimental"])

  def make_graph(self):
    plt.figure()
    plt.plot(self.x, self.precision, "p-", label="precision")
    plt.plot(self.x, self.recall, "p-", label="recall")
    plt.xlabel(self.xlabel)
    plt.title( f"CMS {self.title} precision/recall" )
    plt.legend()
    plt.figure()
    plt.title( f"CMS {self.title} 2n/b threshold" )
    ax = plt.gca() 
    ax.set_xlabel(self.xlabel)
    ax.set_ylabel("probability", color='tab:blue')
    ax.plot(self.x, self.pct_50_prob_theoretical, "p-", label=">2n/b prob (theory)")
    ax.plot(self.x, self.pct_50_prob_experimental, "p-", label=">2n/b prob (exp)")
    ax.tick_params(axis='y', labelcolor="tab:blue")
    ax.legend(loc=1)
    ax2 = ax.twinx()
    ax2.set_ylabel("2n/b threshold", color='tab:red')
    ax2.plot(self.x, self.pct_50_threshold, "p-", color="tab:red", label="2n/b threshold")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax2.legend(loc=3)
    plt.figure()
    plt.plot(self.x, self.overcnt_theoretical, "p-", label="theoretical")
    plt.plot(self.x, self.overcnt_experimental, "p-", label="experimental")
    plt.xlabel(self.xlabel)
    plt.ylabel("overcount")
    plt.title( f"CMS {self.title} n/b overcount" )
    plt.legend()

if __name__ == "__main__":
  random.seed()
  k = 200 # Top 200 cities with reported accidents
  buckets = [800, 1200, 1600, 2400, 3200] # bucket size
  #buckets = [400, 1600]
  buckets_constant = 1600
  hashes = [1,3,5,7,10]   # number of hashes
  #hashes = [1,5]
  hashes_constant = 5
  
  # run over buckets
  bucket_graph = Graph(buckets, "b (buckets)", f"h={hashes_constant}")
  for b in buckets:  
    cms = CountMinSketch(k=k, buckets=b, hashes=hashes_constant)
    st = time.time()
    # Data is a list of city/state where incidient occured
    for line in open('testdata.txt'):
      cms.process(line)

    result = cms.analyze()
    bucket_graph.append_result(result)
    print(f"buckets: {b}, hashes: {hashes_constant}, k: {k}")
    print("runtime", time.time() - st) 

  bucket_graph.make_graph()

  # run over hashes
  hashes_graph = Graph(hashes, "h (hashes)", f"b={buckets_constant}")
  for h in hashes:  
    cms = CountMinSketch(k=k, buckets=buckets_constant, hashes=h)
    st = time.time()
    # Data is a list of city/state where incidient occured
    for line in open('testdata.txt'):
      cms.process(line)

    result = cms.analyze()
    hashes_graph.append_result(result)
    print(f"buckets: {buckets_constant}, hashes: {h}, k: {k}")
    print("runtime", time.time() - st) 

  hashes_graph.make_graph()

  # generate overcnt distribution at b=800 and h=5
  # print("Generate overcnt distribution")
  st = time.time()
  cms = CountMinSketch(k=k, buckets=buckets_constant, hashes=hashes_constant)
  for line in open('testdata.txt'):
    cms.process(line)

  result = cms.analyze()
  print(time.time() - st)
  
  plt.figure()
  ax = plt.gca()
  plt.hist(result["cnt_dist"], color='blue', edgecolor='black',
    bins=int(len(result["cnt_dist"])/200)) 
  plt.xlabel( "distribution of count" )
  plt.title( f"Precise Count Distribution" )
  nk_en = cms.n/cms.k - result["pct_50_threshold"]
  plt.axvline( nk_en, color="red")
  plt.text(nk_en, ax.get_ylim()[1]-4, f"nk-en ({nk_en})",
    horizontalalignment='center', verticalalignment='center', color="red",
    bbox=dict(facecolor='white', alpha=0.9))
  
  plt.figure()
  ax = plt.gca()
  plt.hist(result["overcnt_dist"], color='blue', edgecolor='black',
    bins=len(result["overcnt_dist"])) 
  plt.xlabel( "distribution of overcount" )
  plt.title( f"CMS overcount distribution h={hashes_constant} b={buckets_constant}" )
  plt.axvline(result["pct_50_threshold"], color="red")
  threshold_pct= 1 - result["pct_50_prob_experimental"]
  plt.text(result["pct_50_threshold"], ax.get_ylim()[1]-4, f"{threshold_pct*100:.5f}%",
    horizontalalignment='center', verticalalignment='center', color="red",
    bbox=dict(facecolor='white', alpha=0.9))
  plt.show()
