
import numpy as np
import matplotlib.pyplot as plt

N_samples = 10000

mu1 = 100.
mu2 = 100.
sigma1 = 20.
sigma2 = 10.
h = 2

def volume(a1, a2, h):
  return h*(a1 + a2 + np.sqrt(a1*a2))/3

def compute_mu(mu1, mu2, h):
  return volume(mu1, mu2, h)

def compute_sigma(sigma1, sigma2, h):
  return h*np.sqrt(sigma1**2+sigma2**2)/3

def compute_sigma_orig(sigma1, sigma2, h):
  return h*(sigma1+sigma2)/3


# sample X and Y
s1 = np.random.normal(mu1, sigma1, N_samples)
s2 = np.random.normal(mu2, sigma2, N_samples)

# Compute Z
v = volume(s1, s2, h)

# Plot hist of Z
count, bins, ignored = plt.hist(v, 100, normed=True)

# Compute and plot approx of Z
mu = compute_mu(mu1, mu2, h)
sigma = compute_sigma(sigma1, sigma2, h)
sigma_orig = compute_sigma_orig(sigma1, sigma2, h)

plt.plot(
    bins,
    1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    linewidth=2, color='r')

plt.plot(
    bins,
    1/(sigma_orig * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma_orig**2) ),
    linewidth=2, color='g')
plt.show()

