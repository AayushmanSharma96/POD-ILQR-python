'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Python class for linearized system identification through MujoCo simulator

Date: July 5, 2019
'''
#!/usr/bin/env python

import numpy as np
import scipy.linalg.blas as blas
#from PFM import *
from material_params import *
#from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
# import burgers_test as simulator
import burgers as simulator
import sys
import PFM
# import PFM_CH



class ltv_sys_id_class(object):

	def __init__(self, model_xml_string,  state_size, action_size, n_substeps=1, n_samples=500):

		self.n_x = state_size
		self.n_u = action_size

		# Standard deviation of the perturbation 
		self.sigma = 1e-7
		self.n_samples = n_samples

		#self.sim = ACsolver()#MjSim(load_model_from_path(model_xml_string), nsubsteps=n_substeps)
		


	def sys_id(self, x_t, u_t, central_diff, activate_second_order=0, V_x_=None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u
		n_z = np.shape(C)[0]
		# print('Shape of X_t = ', np.shape(x_t))

		##########################################################################################
		
		XU = np.random.normal(0.0, self.sigma, (self.n_samples, n_x + n_u))
		X_ = np.copy(XU[:, :n_x])
		U_ = np.copy(XU[:, n_x:])

		Cov_inv = (XU.T) @ XU
		V_x_F_XU_XU = None

		if central_diff:
			
			F_X_f = simulate((x_t.T) + X_, (u_t.T) + U_)
			F_X_b = simulate((x_t.T) - X_, (u_t.T) - U_)
			Y = 0.5*(F_X_f - F_X_b)
			
		else:

			Y = (simulate((x_t.T) + X_, (u_t.T) + U_) - simulate((x_t.T), (u_t.T)))

			
		# print('Y = '+str(np.shape(Y)))
		F_XU = np.linalg.solve(Cov_inv, (XU.T @ Y)).T

		ZU = np.random.normal(0.0, self.sigma, (self.n_samples, n_z + n_u))
		# XU[:,1: n_x-1] = 0
		Z_ = np.copy(ZU[:, :n_z])
		U_Z = np.copy(ZU[:, n_z:])
		# ZU = np.hstack([Z_, U_Z])

		Z_new = Z_ @ C#np.zeros((self.n_samples, n_x))
		# Z_new[:,0] = Z_[:,0]
		# Z_new[:, -1] = Z_[:,1] 


		# print('test = ',np.linalg.norm(Z_new - Z_ @ C))

		Cov_inv_Z = (ZU.T) @ ZU
		
		Y_Z = (simulate((x_t.T) + Z_new, (u_t.T) + U_Z) - simulate((x_t.T), (u_t.T)))
		Y_Z = Y_Z @ C.T

		F_ZU = np.linalg.solve(Cov_inv_Z, (ZU.T @ Y_Z)).T
		F_x = np.copy(F_XU[:, 0:n_x])
		F_z = np.copy(F_ZU[:, 0:n_z])
		print('F_XU = ',np.round(F_XU,2))
		print('F_ZU = ', np.round(F_ZU,2))
		# print(C @ F_x @ C.T-F_z)#np.linalg.norm((C @ F_x) @ C.T-F_z))

		# print(np.shape(Y @ C.T))
		sys.exit()
		#print('sim = '+str(np.shape(simulate((x_t.T), (u_t.T)))))

		#A = F_XU[:, 0:state_dimension]
		#B = F_XU[:, state_dimension:]
		'''print(np.shape(Y))
		print(np.shape(F_XU @ XU.T))
		diff = (F_XU @ XU.T).T - Y
		diff2 = diff @ diff.T
		print('DIFF = ', np.mean(diff2))'''


		# If the second order terms in dynamics are activated as in the original DDP (not by default)
		if activate_second_order:

			assert (central_diff == activate_second_order)
			assert V_x_ is not None

			
			Z = (F_X_f + F_X_b - 2 * simulate((x_t.T), (u_t.T))).T

			D_XU = self.khatri_rao(XU.T, XU.T)
			
			triu_indices = np.triu_indices((n_x + n_u))
			linear_triu_indices = (n_x+n_u)*triu_indices[0] + triu_indices[1]
			
			D_XU_lin = np.copy(D_XU[linear_triu_indices,:])
			V_x_F_XU_XU_ = np.linalg.solve(10**12 * D_XU_lin @ D_XU_lin.T, 10**(12)*D_XU_lin @ (V_x_.T @ Z).T)
			D = np.zeros((n_x+n_u, n_x+n_u))
			# for ind, v in zip(list(np.array(triu_indices).T), V_x_F_XU_XU_):
			# 	V_x_F_XU_XU[ind] = v
			j=0
			for ind in np.array(triu_indices).T:
				D[ind[0]][ind[1]] = V_x_F_XU_XU_[j]
				j += 1
			
			V_x_F_XU_XU = (D + D.T)/2
			


		return F_XU, V_x_F_XU_XU	#(n_samples*self.sigma**2)




	def sys_id_FD(self, x_t, u_t, central_diff, activate_second_order=0, V_x_=None):

		'''
			system identification by a forward finite-difference for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u
		##########################################################################################
		
		XU = np.random.normal(0.0, self.sigma, (n_x, n_x + n_u))
		X_ = np.copy(XU[:, :n_x])
		U_ = np.copy(XU[:, n_x:])

		F_XU = np.zeros((n_x, n_x + n_u))
		V_x_F_XU_XU = None

		x_t_next = simulate(x_t.T, u_t.T)

		if central_diff:
			
			for i in range(0, n_x):
				for j in range(0, n_x):

					delta = np.zeros((1, n_x))
					delta[:, j] = XU[i, j]
					
					F_XU[i, j] = (simulate(x_t.T + delta, u_t.T)[:, i] - x_t_next[:, i])/XU[i, j]

			for i in range(0, n_x):
				for j in range(0, n_u):

					delta = np.zeros((1, n_u))
					delta[:, j] = XU[i, n_x + j]
					F_XU[i, n_x + j] = (simulate(x_t.T , u_t.T + delta)[:, i] - x_t_next[:, i])/XU[i, n_x + j]
					
		else:

			Y = (simulate((x_t.T) + X_, (u_t.T) + U_) - simulate((x_t.T), (u_t.T)))

			
	

		return F_XU, V_x_F_XU_XU	





	def simulate(self, X, U):
		
		'''
		Function to simulate a batch of inputs given a batch of control inputs and current states
		X - vector of states vertically stacked
		U - vector of controls vertically stacked
		'''
		################## defining local functions & variables for faster access ################

		#sim = self.sim
		forward_simulate = self.forward_simulate
		state_output = self.state_output

		##########################################################################################
		
		X_next = []

		# Augmenting X by adding a zero column corresponding to time
		X = np.hstack((np.zeros((X.shape[0], 1)), X))

		for i in range(X.shape[0]):

			X_next.append(state_output(forward_simulate(None, X[i] , U[i])))
			
		return np.asarray(X_next)[:,:,0]
	



	def vec2symm(self, ):
		pass




	def forward_simulate(self, sim, x, u):

		'''
			Function to simulate a single input and a single current state
			Note : The initial time is set to be zero. So, this can only be used for independent simulations
			x - append time (which is zero here due to above assumption) before state
		'''
		sdim = int(math.sqrt(state_dimension))
		#sim.set_state_from_flattened(x)
		#sim.forward()
		#sim.data.
		ctrl = np.zeros(self.n_u)
		ctrl[:] = u
		N=sdim#self.n_x#sdim
		# dx = 1/self.n_x
		# dt = 1e-4
		# tperiod = 1000
		# nu = .1#0.01/np.pi
		
		#self.sim.step(x[1:].reshape((sdim, sdim)), np.transpose(ctrl[0:int(len(ctrl)/2)]), np.transpose(ctrl[int(len(ctrl)/2):]), 25)
		#sim.render(mode='window')

		# print(np.shape(x[1:]))
		# return simulator.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], x[1:])

		# return simulator.burgers_1d(x[1:], dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], 10)#.reshape(state_dimension,1)
		# return PFM.simulation(n_ac, 0.001, 0.001, np.transpose(ctrl[0:int(len(ctrl)/2)]).reshape((sdim, sdim)), np.transpose(ctrl[int(len(ctrl)/2):]).reshape((sdim, sdim)), x[1:].reshape((sdim, sdim))).reshape((sdim*sdim, 1))#sim.get_state()
		# return PFM_CH.simulation(n_ch, 0.001, dt_ch, np.transpose(ctrl[0:int(len(ctrl)/2)]).reshape((sdim, sdim)), np.transpose(ctrl[int(len(ctrl)/2):]).reshape((sdim, sdim)), x[1:].reshape((sdim, sdim))).reshape((sdim*sdim, 1))#sim.get_state()
		#return PFM.simulation(25, 0.001, 0.001, np.zeros((sdim, sdim)), np.transpose(ctrl[:]).reshape((sdim, sdim)), x[1:].reshape((sdim, sdim))).reshape((sdim*sdim, 1))
		## GSO ONLY##########
		ctrl_T = self.p_mask*ctrl[0]+self.m_mask*ctrl[1]#+np.random.normal(np.zeros(np.shape(self.p_mask)), process_noise_std)
		ctrl_h = self.p_mask*ctrl[2]+self.m_mask*ctrl[3]
		## GSO ONLY##########
		return PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, x.reshape((sdim, sdim))).reshape((sdim*sdim, 1))
	


	def traj_sys_id(self, x_nominal, u_nominal):	
		
		'''
			System identification for a nominal trajectory mentioned as a set of states
		'''
		
		Traj_jac = []
		
		for i in range(u_nominal.shape[0]):
			
			Traj_jac.append(self.sys_id(x_nominal[i], u_nominal[i]))

		return np.asarray(Traj_jac)
		

	
	def state_output(state):

		pass


	def khatri_rao(self, B, C):
	    """
	    Calculate the Khatri-Rao product of 2D matrices. Assumes blocks to
	    be the columns of both matrices.
	 
	    See
	    http://gmao.gsfc.nasa.gov/events/adjoint_workshop-8/present/Friday/Dance.pdf
	    for more details.
	 
	    Parameters
	    ----------
	    B : ndarray, shape = [n, p]
	    C : ndarray, shape = [m, p]
	 
	 
	    Returns
	    -------
	    A : ndarray, shape = [m * n, p]
	 
	    """
	    if B.ndim != 2 or C.ndim != 2:
	        raise ValueError("B and C must have 2 dimensions")
	 
	    n, p = B.shape
	    m, pC = C.shape
	 
	    if p != pC:
	        raise ValueError("B and C must have the same number of columns")
	 
	    return np.einsum('ij, kj -> ikj', B, C).reshape(m * n, p)
