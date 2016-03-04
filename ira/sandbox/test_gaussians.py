import numpy as np
import matplotlib.pyplot as plt
import data

N_samples = 10000

labels = data.read_labels('/mnt/sda3/data/kaggle-heart/train.csv')
for m in labels.itervalues():

    mu1 = m[0]
    mu2 = m[1]
    print mu1, mu2
    sigma1 = 30.
    sigma2 = 30.
    s1 = sigma1 ** 2
    s2 = sigma2 ** 2
    h = 2


    def piconst(s):
        return np.sqrt(2. * np.pi * s)

    # sample X and Y
    x = np.random.normal(mu1, sigma1, N_samples)
    y = np.random.normal(mu2, sigma2, N_samples)
    z = x * y

    count, bins, ignored = plt.hist(z, 100, normed=True)
    s = s1 * s2 + s1 * mu2**2 + s2*mu1**2
    mu = mu1 * mu2

    pdf_z = 1. / piconst(s) * np.exp(- (bins - mu) ** 2 / (2 * s))

    plt.plot(bins, pdf_z, linewidth=2, color='r')

    plt.show()
