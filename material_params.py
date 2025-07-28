import numpy as np
import math
# import matplotlib.pyplot as plt 

state_dimension = 50*50#6*6
control_dimension = 2*2#state_dimension#2*20*20#2*50*50#200
obs_dimension = 16#state_dimension

# Tuning params for ARMA sys-id
q = 2
q_u = 2

C = np.zeros((obs_dimension, state_dimension))
# C = np.eye(obs_dimension)

# C[0,25*50+5]=1
# C[1,25*50+10]=1
# C[2,25*50+20]=1
# C[3,25*50+30]=1
# C[4,25*50+40]=1
# C[5,25*50+45]=1
# C[0,0]=1
# C[1,20]=1
# C[2,40]=1
# C[3,60]=1
# C[4,80]=1
# C[1,-1]=1
# C[3,32]=1
# C[4,33]=1
# C[5,34]=1
# C[6,35]=1
# C[7,36]=1
# C[8,40]=1
# C[9,state_dimension-1]=1
# C[3,-1]=1
# # C[4,80]=1
# C[2,-1]=1
# for iter in range(obs_dimension):
# 	C[iter, iter*int(state_dimension/obs_dimension)] = 1 
# C[0,0] = 1
# C[1,int(state_dimension/3)] = 1
# C[2,int(2*state_dimension/3)] = 1
# C[3, -1] = 1
# C = np.eye(obs_dimension)
# ranvec = np.random.randint(0, state_dimension, [obs_dimension, ])
# flag = 0
# sdim = int(math.sqrt(state_dimension))
# for iter in range(obs_dimension):
# 	if iter*int(state_dimension/obs_dimension)%sdim==0 and iter is not 0:
# 		flag = 1-flag 
	

# 	if flag==0:
# 		C[iter, int(state_dimension/obs_dimension)*iter] = 1

# 	else:
# 		C[iter, int(state_dimension/obs_dimension)*iter+1] = 1
# iter = 0
# for i in range(10):
# 	C[iter, i*10+4]=1
# 	# C[iter+1, i*10+5]=1
# 	iter+=1
# C[10, 8]=1
iter = 0

for i in range(obs_dimension):
    C[iter, 50*10+3*i] = 1
    iter+=1


# C[1,state_dimension-1] = 1
# C[2,4] = 1
# C[3,6] = 1
# C[4,-1] = 1
#C = np.eye(obs_dimension)

# Cost parameters for nominal design
Q_fs = 9*np.eye(state_dimension)
Q = np.zeros((obs_dimension*q+control_dimension*(q_u-1),obs_dimension*q+control_dimension*(q_u-1)))
Q_final = np.zeros((obs_dimension*q+control_dimension*(q_u-1),obs_dimension*q+control_dimension*(q_u-1)))
Q[:obs_dimension, :obs_dimension] = C @ (Q_fs @ C.T)
Q_final[:obs_dimension, :obs_dimension] = 1000* (C @ (Q_fs @ C.T))
R = .05*2*np.eye(control_dimension)


## Mujoco simulation parameters
# Number of substeps in simulation

horizon = 20#50#800
nominal_init_stddev = 0.1
n_substeps = 5

# Cost parameters for feedback design

W_x_LQR = 10*np.eye(state_dimension)#*state_dimension)
W_u_LQR = 2*np.eye(control_dimension)#*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension)#*state_dimension)


# D2C parameters
feedback_n_samples = 100#2000


# Cahn-Hiliard params
dt_ch=1e-7
n_ch=int(0.01*0.025/dt_ch)

n_ac=25

noise_std_test = 0.0





 # Task.Q = 0*eye(Task.nm*Task.qx+Model.nu*(Task.qu-1));
 #    Task.QT = 0*eye(Task.nm*Task.qx+Model.nu*(Task.qu-1));
 #    Task.Q(1:Task.nm,1:Task.nm) = 1*Task.Ck*[1 0;0 0.1]*Task.Ck';
 #    Task.QT(1:Task.nm,1:Task.nm) = 100*eye(Task.nm);