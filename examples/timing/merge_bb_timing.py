import numpy as np
from sklearn.ensemble import RandomForestRegressor

import dforest
import query

missing_data = lambda x : 100. if x == '?' else float(x)
data = np.loadtxt("./datasets/Auto.data", converters=missing_data, 
                    skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7])

np.random.seed(12345)
np.random.shuffle(data)

auto_X = data[:, 1:]
auto_y = data[:, 0]

model = RandomForestRegressor()
model.fit(auto_X, auto_y)

forest = dforest.make_forest_sklearn(model)
forest_max = forest.copy()
forest_min = forest.copy()

min_bound = np.min(auto_X, axis=0)
max_bound = np.max(auto_X, axis=0)

max_x, _ = query.pso_max(forest, min_bound, max_bound)
min_x, _ = query.pso_min(forest, min_bound, max_bound)

forest_max.merge_max(10, max_x, offset=-2.0)
forest_max.print_summary()

forest.merge_min(10, min_x, offset=1.0)
forest_min.print_summary()

forest.free()
forest_max.free()
forest_min.free()
