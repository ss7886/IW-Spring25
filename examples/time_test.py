import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import dforest

missing_data = lambda x : 100. if x == '?' else float(x)
data = np.loadtxt("../datasets/Auto.data", converters=missing_data, 
                    skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7])

np.random.seed(12345)
np.random.shuffle(data)

auto_X = data[:, 1:]
auto_y = data[:, 0]

model = RandomForestRegressor()
model.fit(auto_X, auto_y)

forest = dforest.make_forest_sklearn(model)

forest.merge(25)
forest.print_summary()
forest.free()