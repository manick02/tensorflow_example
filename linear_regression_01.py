# this is a get to started of linear regression
import numpy as np
import matplotlib.pyplot as plot

num_points = 1000
vectors_set = []
W = 0.1
b = 0.4
for i in range(num_points):
    x1 = np.random.normal(0.0, 1.0)
    # adding noise
    nd = np.random.normal(0.0, 0.05)
    y1 = W * x1 + b
    y1 = y1+nd
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
plot.plot(x_data, y_data, 'r*', label='Original data')
plot.legend()
plot.show()