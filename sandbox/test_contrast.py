import numpy as np
import matplotlib.pyplot as plt

with open('with_contrast.npy', 'rb') as f:
    a = np.load(f)

fig = plt.figure(1)
plt.imshow(a)