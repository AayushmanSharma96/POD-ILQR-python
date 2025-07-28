import numpy as np
from material_params import state_dimension
import pde.burgers_corr as sim
import matplotlib.pyplot as plt

x = np.sin(np.linspace(-2*np.pi, 2*np.pi, 100))

N = 100
dx = (2*np.pi)/N
dt = 1e-4
nu = 0.1#0.01/np.pi
tperiod = 400

# Control at 0 =  [[0.00210698]
#  [0.0019249 ]]
# Control at 1 =  [[-0.00332231]
#  [-0.00498529]]
# Control at 2 =  [[0.00347159]
#  [0.01856356]
x1 = np.load('x_1.npy')
u1 = np.load('u_1.npy')
y1 = sim.simulation(tperiod, nu, dt, u1[0], u1[1], x1)
y2 = sim.simulation(tperiod/4+100, nu, dt, u1[0], u1[1], y1)
# y2 = sim.simulation(tperiod, nu, dt, -0.00332231, -0.00498529, y1)
# y3 = sim.simulation(tperiod, nu, dt, 0.00347159, 0.01856356, y2)
print('y1 = ', y1)
print('y2 = ', y2)
# print('y3 = ', y3)
# plt.plot(np.linspace(-1,1,100), x, 'b')
# plt.plot(np.linspace(-1,1,100), y, 'r')
# plt.show()
# def burgers_1d(u, dx, dt, nu, v_0, v_N, tperiod):
	
# 	# print(np.shape(u[1:]))

# 	# print(np.shape(v_N.reshape(np.shape(u[-1]))))
# 	# print(np.shape(np.r_[u[1:], v_N.reshape(np.shape(u[-1]))]))

# 	U_new = np.zeros((state_dimension, 1))
# 	U_new[:] = u.reshape(state_dimension, 1)
# 	# np.shape(U_new)
# 	# np.shape(U_new[1:])
# 	for t in range(tperiod):
# 		# U_p = np.r_[U_new[1:], v_N.reshape(1, 1)]
# 		# print(np.shape(U_new[1:]))
# 		U_p = np.r_[U_new[1:], v_N.reshape(1, 1)]
# 		U_m = np.r_[v_0.reshape(1, 1), U_new[:-1]]
# 		U_dt =  ((np.multiply(U_new, U_m)-np.multiply(U_new, U_new))/dx + nu*(U_p + U_m - 2*U_new)/(dx*dx))
# 		# U_dt =  (((U_new @ U_m)-(U_new @ U_new))/dx + nu*(U_p + U_m - 2*U_new)/(dx*dx))
# 		# print('test size = ',np.shape(U_new[1:]))
# 		# U_new = u + dt*U_dt
# 		U_new = U_new + dt*U_dt

# 	# print(np.multiply(U_new, U_m))

# 	return U_new