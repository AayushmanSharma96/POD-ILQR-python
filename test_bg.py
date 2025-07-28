import numpy as np
# import burgers_test as sim
import burgers as sim
import math

dx = 0.01
dt = 0.0001
nu = 0.1#1/np.pi 
N = int(1/dx)

x = np.zeros([int(1/dx),1])
x[:] = np.random.normal(0, 1, [int(1/dx), 1])
print(x[:].T)


# for t in range(3):
# u = sim.burgers_1d(u, dx, dt, nu, np.array([1.0]), np.array([-2.0]), 50)
x = sim.simulation(300, nu, dt, np.array([1.0]), np.array([-2.0]), x)
y = np.array([x[0], x[-1]])
print(x[:].T)
print(y)