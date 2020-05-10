import numpy as np



# np.load()
predicted = np.random.normal(5, 0.1, (1, 64, 64, 313))
t = 0.38

num = np.exp(np.log(predicted)/t)
denom = np.sum(num, axis=3)
result = num/denom






