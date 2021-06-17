from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np
n_samples = 500
X, y = datasets.make_regression(n_samples=n_samples,
                                n_features=1,
                                n_informative=2,
                                noise=5,
                                random_state=0)  # 2)
n_outliers = 100
X[:n_outliers], y[:n_outliers] = datasets.make_regression(n_samples=n_outliers,
                                                          n_features=1,
                                                          n_informative=2,
                                                          noise=2,
                                                          random_state=61)
y = np.expand_dims(y, axis=1)
plt.scatter(X[:], y[:], marker='.')
print(np.shape(X))
print(np.shape(y))
