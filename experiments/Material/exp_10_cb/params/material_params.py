'''
Inverted pendulum parameters

'''
import numpy as np


state_dimension = 20*20#50*50#100
control_dimension = 2*state_dimension#2*20*20#2*50*50#200

# Cost parameters for nominal design
Q = 9*np.eye(state_dimension)#*state_dimension)
Q_final = 9000*np.eye(state_dimension)#*state_dimension)
R = .05*2*np.eye(control_dimension)#2*control_dimension*control_dimension)

## Mujoco simulation parameters
# Number of substeps in simulation

horizon = 100#50#800
nominal_init_stddev = 0.1
n_substeps = 5

# Cost parameters for feedback design

W_x_LQR = 10*np.eye(state_dimension)#*state_dimension)
W_u_LQR = 2*np.eye(control_dimension)#*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension)#*state_dimension)


# D2C parameters
feedback_n_samples = 2000