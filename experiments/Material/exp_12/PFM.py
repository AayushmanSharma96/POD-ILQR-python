import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
#import shutil
from params.material_params import state_dimension, control_dimension
import math





class ACsolver():

	def __init__(self):
		self.state = np.zeros([state_dimension, state_dimension]).reshape((state_dimension*state_dimension, ))		
	


	def step(self, phi0, T, h, n_step):

		N_nodes_edge=int(math.sqrt(state_dimension))
		N_macro_grid_edge=int(math.sqrt(control_dimension/2))

		Nx = N_nodes_edge
		Ny = N_nodes_edge
		NxNy = Nx*Ny
		
		dx = 0.1
		dy = 0.1
		x=(np.arange(Nx))*dx
		y=(np.arange(Ny))*dy
		X,Y = np.meshgrid(x,y)

		t_time = 0.0
		d_time=0.001#0.00005
		#n_step=2000#200001
		n_print=10#499#4000
		n=0.0

		T = T.reshape((N_macro_grid_edge, N_macro_grid_edge))
		h = h.reshape((N_macro_grid_edge, N_macro_grid_edge))
		if Nx is not N_macro_grid_edge:
			T1=np.repeat(T,Nx/N_macro_grid_edge,axis=1)
			T=np.repeat(T1,Ny/N_macro_grid_edge,axis=0)
			h1=np.repeat(h,Nx/N_macro_grid_edge,axis=1)
			h=np.repeat(h1,Ny/N_macro_grid_edge,axis=0)

		# Controlling parameters
		M=0.01

		# Initial condition
		
		#phi0=np.loadtxt('data.txt', dtype=float, delimiter='\t')
		phi=phi0.copy()
		Fts=[]
		time=[]
		ims=[]


		for n in range(n_step):
			t_time += d_time
			phiXP = vstack((phi[1:, :], phi[0, :]))
			phiXM = vstack((phi[-1, :], phi[0:-1, :]))
			phiYP = hstack((phi[:, 1:], phi[:, 0][:, newaxis]))
			phiYM = hstack((phi[:, -1][:, newaxis], phi[:, 0:-1]))

			phi = phi + d_time * (-4 * phi * phi * phi -2 * T * phi + h + M * ((phiXP - 2 * phi + phiXM) / dx ** 2 + (phiYP - 2 * phi + phiYM) / dy ** 2))


			phi[phi>0.9999] = 0.9999
			phi[phi<-0.9999]=-0.9999
			
			
			time.append(t_time)

			'''if n % n_print == 0:
				fig = plt.figure(figsize=(10, 6.5))
				ax = fig.add_subplot(121.5)
				CS = ax.contourf(X, Y, phi, cmap=plt.cm.get_cmap('RdBu'))# v, cmap=plt.cm.get_cmap('RdBu'))
				ax.set_title('phase field evolution')
				ax.set_xlabel('x')
				ax.set_ylabel('y')
				ax.set_aspect(1.0)

				axs=[ax]
				fig.suptitle('Explicit | 100x100', position=(0.2, 0.94), fontsize=20, fontweight='bold')
				if t_time <= 5:
					fig.text(0.15, 0.84, 'T = -2, h = 0', fontsize=16)
				else:
					fig.text(0.02, 0.84, 'T = rand(-2,2), h = rand(-1,0)', fontsize=16)
				fig.text(0.35, 0.915, 'Time = %2.2f sec' % t_time, fontsize=16, bbox={'facecolor': 'yellow', 'alpha': 0.3})

				ims.append([CS.collections,])

				idx=int(n/n_print)
				savefile='./file' + str(idx) + '.png'
				plt.savefig(savefile, dpi=300)
				plt.draw()
				plt.pause(0.02)
				plt.close()'''
		self.state = phi.copy().reshape([state_dimension, 1])

		return phi.reshape([state_dimension, 1])

	def get_state(self):
		return self.state

	def set_state(self, new_state):
		self.state = new_state

	def plotCMAP(self, phi, t_time):
		ims = []
		x=(np.arange(int(sqrt(state_dimension))))*0.1
		y=(np.arange(int(sqrt(state_dimension))))*0.1
		X,Y = np.meshgrid(x,y)
		fig = plt.figure(figsize=(10, 6.5))
		ax = fig.add_subplot(121.5)
		CS = ax.contourf(X, Y, phi, cmap=plt.cm.get_cmap('RdBu'))# v, cmap=plt.cm.get_cmap('RdBu'))
		ax.set_title('phase field evolution')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_aspect(1.0)
		axs=[ax]
		fig.suptitle('Explicit | 10x10', position=(0.2, 0.94), fontsize=20, fontweight='bold')
		#if t_time <= 5:
		#	fig.text(0.15, 0.84, 'T = -2, h = 0', fontsize=16)
		#else:
		#	fig.text(0.02, 0.84, 'T = rand(-2,2), h = rand(-1,0)', fontsize=16)
			
		fig.text(0.35, 0.915, 'Step = %f' % t_time, fontsize=16, bbox={'facecolor': 'yellow', 'alpha': 0.3})
				
		ims.append([CS.collections,])

		idx=int(t_time)
		savefile='/home/karthikeya/D2C-2.0/experiments/Material/exp_2/Images/file' + str(idx) + '.png'
		plt.savefig(savefile, dpi=300)
		plt.draw()
		plt.pause(0.002)
		plt.close()

'''T = np.ones([Nx, Ny])
a = ACsolver(T, h)
V = a.step(phi0, T, h, 20000)
V2 = a.step(phi0, 200*T, h, 20000)
V3 = a.step(phi0, 20000*T, h, 20000)
print(sum((V)))
print(sum((V2)))
print(sum((V3)))
print('T = ', T)'''

