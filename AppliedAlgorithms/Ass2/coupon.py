#!/usr/bin/python

import random
import math
import matplotlib.pyplot as plt

class CouponCollector():
  def __init__(self, N):
    """
    CouponCollector simulator
    N - number of coupons
    """
    self.N = N
    self.found = []

  def findNextCoupon(self):
    # technically should return error, but meh
    unfound_coupons = self.N - len(self.found)
    if unfound_coupons <= 0:
      return 0

    trials = 0
    coupon = None
    while coupon is None or coupon in self.found:
      trials += 1
      coupon = int(random.uniform(0, self.N))
    self.found.append(coupon)
    
    return trials

if __name__ == "__main__":
  random.Random(None)

  # number of couples to try
  N = [10, 50, 100, 200, 500, 1000]
  samples = 100
  N_select = [10, 50, 100] # plot x_n_k as well for these N

  experimental_trials = []
  theoretical_trials = []
  
  plt.figure()
  for n in N:
    total_trials_experimental = 0
    # theoretical number of trials
    total_trials_theoretical = n * math.log(n) + 0.57 * n
    theoretical_trials.append(total_trials_theoretical)
    # random variable: number of trials needed to get a new coupon when k remains out of n
    samples_remain = [ n-i for i in range(n) ]
    x_n_k_theoretical = [ n/k for k in samples_remain ]
    x_n_k_experimental = [0] * n
    for _ in range(samples):
      c = CouponCollector(n)
      for i in range(n):
        experimental = c.findNextCoupon()
        total_trials_experimental += experimental
        if n in N_select:
        	x_n_k_experimental[i] += experimental 
    for i in range(n):
      x_n_k_experimental[i] = x_n_k_experimental[i] / samples
    avg_experimental_trials = total_trials_experimental / samples
    experimental_trials.append(avg_experimental_trials)
    # Plot the x_n_k
    if n in N_select:
      plt.plot(samples_remain, x_n_k_theoretical, "-", label=f"theoretical-N={n}")
      plt.plot(samples_remain, x_n_k_experimental, "x", label=f"experimental-N={n}")
      plt.ylabel('number of trials')
      plt.xlabel(f'k coupons remaining of N')
      plt.legend()
  # Plot the experimental vs theoretical
  plt.figure()
  plt.plot(N, experimental_trials, "p", label="experimental")
  plt.plot(N, theoretical_trials, "-", label="theoretical")
  plt.ylabel('number of trials')
  plt.xlabel('N')
  plt.legend()
  plt.show()
