import numpy as np
from sklearn.ensemble import RandomForestRegressor

import dforest
import smt_query

missing_data = lambda x : 100. if x == '?' else float(x)
data = np.loadtxt("../datasets/Auto.data", converters=missing_data, 
                    skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7])

np.random.seed(12345)
np.random.shuffle(data)
np.set_printoptions(precision=1, suppress=True)

auto_X = data[:, 1:]
auto_y = data[:, 0]

model = RandomForestRegressor()
model.fit(auto_X, auto_y)

x = np.array([6, 225.0, 100.0, 3233, 15.4, 76, 1])
delta = np.array([2, 20, 10, 100, 1.0, 1, 2])
y = model.predict([x])[0]
epsilon = 5
print(smt_query.query(model, x - delta, x + delta, y - epsilon, y + epsilon))
