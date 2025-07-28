import numpy as np
import control

A = np.load('A_test.npy')

n = np.shape(A)[0]
state_dimension = n
obs_dimension = int(n/5)

C = np.zeros((obs_dimension, state_dimension))
for iter in range(obs_dimension):
	C[iter, int(state_dimension/obs_dimension)*iter] = 1

# y = np.shape(C @ A)
# O = np.zeros((n*y[0], y[1]))

# for i in range(n):
# 	O[i*y[0]:(i+1)*y[0],:] = C @ (A**i)
# print(np.linalg.matrix_rank(O)==n)

Z = control.obsv(A,C)
print(np.linalg.matrix_rank(Z))