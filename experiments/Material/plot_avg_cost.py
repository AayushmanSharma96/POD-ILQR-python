import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Declare the `path_to_model_free_DDP` as the path to the main folder, before proceeding
path_to_model_free_DDP="/home/karthikeya/D2C-2.0"
path_to_folder = path_to_model_free_DDP+"/experiments/Material"


with open(path_to_folder+"/exp_11/training_cost_data.txt", 'r') as f:
	J_1 = np.array([float(line.strip()) for line in f])
	#J_1 = J_1/J_1[-1]

with open(path_to_folder+"/exp_12/training_cost_data.txt", 'r') as f:
	J_2 = np.array([float(line.strip()) for line in f])
	#J_2 = J_2/J_2[-1]
'''with open(path_to_folder+"/exp_9v3/training_cost_data.txt", 'r') as f:
	J_3 = np.array([float(line.strip()) for line in f])
	J_3 = J_3/J_3[-1]
with open(path_to_folder+"/exp_9v4/training_cost_data.txt", 'r') as f:
	J_4 = np.array([float(line.strip()) for line in f])
	J_4 = J_4/J_4[-1]
with open(path_to_folder+"/exp_9v5/training_cost_data.txt", 'r') as f:
	J_5 = np.array([float(line.strip()) for line in f])
	J_5 = J_5/J_5[-1]'''
J = [J_1, J_2]#, J_3, J_4, J_5]

J_avg = np.mean(J, axis=0)
J_std = np.std(J, axis=0)

#mpl.style.use('default')
mpl.style.use('seaborn')

# Plot the mean and average of the above data w.r.t iterations as follows:
#plt.plot(np.array([*range(0, J_avg.shape[0])]), J_avg, linewidth=3)
plt.plot(np.array([*range(0, J_1.shape[0])]), J_1, linewidth=3)
plt.plot(np.array([*range(0, J_2.shape[0])]), J_2, linewidth=3)
#plt.fill_between(np.array([*range(0, J_avg.shape[0])]), J_avg-J_std, J_avg+J_std, color='C5', alpha=0.5)
plt.xlabel("Training iteration count", fontweight="bold", fontsize=20)
plt.xticks(fontsize=14)
plt.ylabel("Average episodic cost fraction", fontweight="bold", fontsize=20)
plt.yticks(fontsize=14)
plt.grid(color='white', linewidth=1.5)
#plt.legend(['Mean', 'Standard deviation'], fontsize=20)
plt.legend(['Initial U_{mean}=0', 'Initial U_{mean}=single-pt'], fontsize=20)
#plt.title("Episodic cost (avg. of 5 trials) vs No. of training iterations", fontsize=15)
plt.show()
