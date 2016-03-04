import data
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split


labels = data.read_labels('train.csv')


x = [(e[0], e[1][0]) for e in labels.iteritems()]
y = [e[1][1] for e in labels.iteritems()]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

p_train, x_train = zip(*x_train)
p_test, x_test = zip(*x_test)

# clf = RandomForestRegressor(n_estimators=10000, max_depth=, n_jobs=-1)
clf = Ridge(alpha=0.3)

fff = lambda e: [e, np.log(e)]

clf.fit([fff(e) for e in x_train], y_train)


mse = []

for a, b, c in zip(clf.predict([fff(e) for e in x_test]), y_test, p_test):
    print c, a, b
    mse.append((a-b)**2)

print np.mean(mse)
print np.sqrt(np.mean(mse))


import matplotlib.pyplot as plt


N = 50


plt.scatter(x_test, y_test, alpha=0.5)
plt.show()
