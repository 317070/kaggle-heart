
import numpy as np
import matplotlib.pyplot as plt

N_samples = 100000

mu1 = 400.
mu2 = 300.
sigma1 = 10
sigma2 = 40
h = .4

def volume(a1, a2, h):
  return h*(a1*a2)*np.pi/4.0

def compute_mu(mu1, mu2, h):
  return volume(mu1, mu2, h)

def compute_sigma(sigma1, mu1, sigma2, mu2, h):
  return h*np.sqrt(sigma1**2*sigma2**2 + sigma1**2*mu2**2 + sigma2**2*mu1**2)*np.pi/4
# sample X and Y
s1 = np.random.normal(mu1, sigma1, N_samples)
s2 = np.random.normal(mu2, sigma2, N_samples)

# Compute Z
v = volume(s1, s2, h)

# Plot hist of Z
count, bins, ignored = plt.hist(v, 100, normed=True)

# Compute and plot approx of Z
mu = compute_mu(mu1, mu2, h)
sigma = compute_sigma(sigma1, mu1, sigma2, mu2, h)
print sigma

plt.plot(
    bins,
    1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    linewidth=2, color='r')

plt.show()

