
import numpy as np
import matplotlib.pyplot as plt

N_samples = 10000

mu1 = 200.
mu2 = 100.
sigma1 = 10.
sigma2 = 10.
h = 2

def volume(a1, a2, h):
  return h*(a1 + a2 + np.sqrt(a1*a2))/3

def compute_mu(mu1, mu2, h):
  return volume(mu1, mu2, h)

def compute_sigma(sigma1, sigma2, h):
  return h*(sigma1 + sigma2)/3

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

plt.plot(
    bins,
    1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
    linewidth=2, color='r')
#plt.show()


## layer in numpy:
def jeroenLayer(mu_area, sigma_area, is_not_padded, slicelocs):
  """

  Args:
    All arguments are N x M matrices, where N is the batch size and M the number of slices.
  """
  # For each slice pair, compute if both of them are valid
  is_pair_not_padded = is_not_padded[:, :-1] + is_not_padded[:, 1:] > 1.5
  # Compute the distance between slices
  h = np.abs(slicelocs[:, :-1] - slicelocs[:, 1:])
  # Compute mu for each slice pair
  m1 = mu_area[:, :-1]
  m2 = mu_area[:, 1:]
  mu_volumes = (m1 + m2 + np.sqrt(m1*m2)) * h / 3.0
  mu_volumes = mu_volumes * is_pair_not_padded
  # Compute sigma for each slice pair
  s1 = sigma_area[:, :-1]
  s2 = sigma_area[:, 1:]
  sigma_volumes = h*(s1 + s2) / 3.0
  sigma_volumes = sigma_volumes * is_pair_not_padded
  # Compute mu and sigma per patient
  mu_volume_patient = np.sum(mu_volumes, axis=1)
  sigma_volume_patient = np.sqrt(np.sum(sigma_volumes**2))

  return mu_volume_patient, sigma_volume_patient

def test_jeroenlayer():
  mu_area1       = np.array([1, 1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 1])
  sigma_area1    = np.array([3, 3, 3, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2])
  is_not_padded1 = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
  slicelocs1     = np.array([1, 1, 1, 3, 4, 6, 7, 7, 7, 8, 9, 9, 9])

  # compute by hand:
  sqrt = np.sqrt
  v1 = 2./3.*(3+sqrt(2))
  v2 = 1./3.*(5+sqrt(6))
  v3 = 2./3.*(7+sqrt(12))
  v4 = 1./3.*(9+sqrt(20))
  v5 = 0
  v6 = 0
  v7 = 1./3.*(5+sqrt(6))
  v8 = 1./3.*(3+sqrt(2))
  volumes1       = np.array([0, 0, v1, v2, v3, v4, v5, v6, v7, v8, 0, 0])
  s1 = 4 * 2./3.
  s2 = 2 * 1./3.
  s3 = 3 * 2./3.
  s4 = 4 * 1./3.
  s5 = 3 * 0./3.
  s6 = 2 * 0./3.
  s7 = 2 * 1./3.
  s8 = 3 * 1./3.
  sigmas1        = np.array([0, 0, s1, s2, s3, s4, s5, s6, s7, s8, 0, 0])
  s_tot = sqrt(np.sum(sigmas1*sigmas1))
  v_tot = np.sum(volumes1)

  print s_tot
  print v_tot
  print jeroenLayer(np.array([mu_area1]), np.array([sigma_area1]), np.array([is_not_padded1]), np.array([slicelocs1]))


test_jeroenlayer()