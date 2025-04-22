import numpy as np
from sklearn.ensemble import RandomForestRegressor

import dforest
import query

missing_data = lambda x : 100. if x == '?' else float(x)
data = np.loadtxt("./datasets/Auto.data", converters=missing_data, 
                    skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7])

np.random.seed(12345)
np.random.shuffle(data)
np.set_printoptions(precision=1, suppress=True)

auto_X = data[:, 1:]
auto_y = data[:, 0]

model = RandomForestRegressor()
model.fit(auto_X, auto_y)

forest = dforest.make_forest_sklearn(model)

x = np.array([6, 225.0, 100.0, 3233, 15.4, 76, 1])
delta = np.array([2, 40, 18, 250, 1.8, 1, 2])
champ_x, champ_y = query.pso_max(forest, x - delta, x + delta)
print(f"PSO Max:")
print(f"    Champ x:{champ_x}, Champ y: {champ_y}")
champ_x, champ_y = query.pso_min(forest, x - delta, x + delta)
print(f"PSO Min:")
print(f"    Champ x:{champ_x}, Champ y: {champ_y}")

forest.free()
