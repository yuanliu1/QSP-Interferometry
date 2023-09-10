from cvqi import cvqIon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from qutip import *
from qutip.measurement import measure
import numpy as np

# Initialize ion for QSPI-sensing Wigner plot using cvqIon class from cvqi.py
mass = 1  # Mass of ion
q = 1  # Ion charge
N = 400  # Size of motional Hilbert space or number of Fock states
w_axial = 1  # Trap axial secular frequency
modes = 1  # Number of motional modes
psi_0 = tensor(fock(N, 0), (basis(2, 0)).unit()) # ground state for a cvqIon (qubit+oscillator system)
ion = cvqIon(mass, q, w_axial, N, modes, psi_0) # initialize cvqIon (qubit+oscillator system) in ground state
mode = 0 # mode of the qubit+oscillator on which to act

Z0, Z1 = ket2dm(basis(2, 0)), ket2dm(basis(2, 1)) # projection matrices for qubit states
PZ1 = [tensor(qeye(N), Z0), tensor(qeye(N), Z1)] # projection matrices for qubit states with oscillator included

# set the scale parameters and the selected degree d
k = 1 / np.sqrt(2)
d = 9

# Ex. angle set
# thresh = k * np.pi / 4
# d = 9
angles = [3.96027657, 3.3228269, 0.58410433, 0.67308454, 3.25323775, 2.61830191, 2.86563178, 3.8736934, 5.22492424, 2.92001249, 3.09553139, 3.16709671, 3.5727264, 0.77114436, 3.38987993, 1.57901798] # learned phases for d = 9

# Set size of Wigner plot in x and y
xvec = np.linspace(-20, 20, 2500)
yvec = np.linspace(-20, 20, 2500)

d = len(angles) - 1 # calculate the degree d corresponding to the number of phase angles in the sequence
ion.carrier_evo(mode, 0, 2 * angles[0], 1) # simulate the initial specified rotation for QSP
for i in range(d):
    ion.cntrl_disp_evo(mode, k, 1) # simulate the controlled displacements by k
    ion.carrier_evo(mode, 0, 2 * angles[i + 1], 1) # simulate the rotations by the specified phase angles
    # There is a factor of 2 because of our convention for qubit rotations in the CVQIon class

psi = ion.history_without_noise # extract ion history after simulating the 

# Plot
print("Making plot")

cat = (coherent(N, d * k * np.sqrt(2) * 1j) + coherent(N, -d * k * np.sqrt(2) * 1j)).unit() # construct a cat state of equivalent displacement magnitude

fig, ax1 = plt.subplots(1, 1, figsize = (10, 10)) # create the figure
plt.rcParams.update({'font.size': 48}) # update fontsize for plots
plt.rc('font', size = 48) # fontsize of the plot
plt.rc('axes', titlesize = 24)     # fontsize of the axes title
plt.rc('axes', labelsize = 24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = 24)    # fontsize of the tick labels
plt.rc('ytick', labelsize = 24)    # fontsize of the tick labels

W_F = wigner((tensor(qeye(N), create(2)) * psi[-1]).ptrace(0).unit(), xvec, yvec) # code for generating Wigner plot of upper-left matrix entry F
# W_G = wigner((tensor(qeye(N), destroy(2)) * psi[-1]).ptrace(0).unit(), xvec, yvec) # code for generating Wigner plot of upper-right matrix entry G
# W_cat = wigner(cat, xvec, yvec) # code for generating Wigner plot of cat state with same displacement
  
wlim = abs(W_F).max() # getting maximum value in Wigner plot to set scale

# save Wigner plot data for future processing
filename_wigner = '20230906_qspi_d_9_k_1_size_40_res_2500_N_400.npy' # use convention of saving with date, type of state, d, k, size, resolution, and number of Fock levels
with open(filename, 'wb+') as f:
    np.save(f, W_F)

# make the desired Wigner plot, using an appropriate number of colors in the color map
ax1.contourf(xvec, yvec, W_F, 1000, norm = mpl.colors.Normalize(-wlim, wlim), cmap = mpl.cm.get_cmap('RdBu'))
ax1.set_title('QSPI Sensing State d = ' + str(d), fontsize = 48)
plt.draw()

# ax1.contourf(xvec, xvec, W_G, 1000, norm = mpl.colors.Normalize(-wlim, wlim), cmap = mpl.cm.get_cmap('RdBu'))
# ax1.set_title('QSPI Sensing State G d = ' + str(d), fontsize = 48)
# plt.draw()

print("Finished plot")
plt.show()

# save the Wigner plot itself as a png file
save_title = "20230906_qspi_d_9_k_1_size_40_res_2500_N_400" # use same titling convention as with npy file
fig.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight')

##################################################################################################################
# Processing of Wigner function data

xvec = np.linspace(-3, 3, 2400)
yvec = np.linspace(-3, 3, 2400)

W_qsp = np.load("20230908_sensing_wigner_title_d_9_k_1_sqrt2_size_6_res_2400_N_400.npy")
marginalx_qsp = np.trapz(W_qsp, xvec, axis = 0)
marginalp_qsp = np.trapz(W_qsp, yvec, axis = 1)
wlim_qsp = W_qsp.max()

W_cat = np.load("20230908_cat_state_9i_corrected_title_d_9_k_1_sqrt2_size_6_res_2400_N_400.npy")
marginalx_cat = np.trapz(W_cat, xvec, axis = 0)
marginalp_cat = np.trapz(W_cat, yvec, axis = 1)
wlim_cat = W_cat.max()

fig2 = plt.figure(figsize = (8, 8))
gs2 = gridspec.GridSpec(3, 3)
ax_main2 = plt.subplot(gs2[1:3, :2])
ax_xDist2 = plt.subplot(gs2[0, :2], sharex = ax_main2)
ax_pDist2 = plt.subplot(gs2[1:3, 2],sharey = ax_main2)

ax_main2.contourf(xvec, yvec, W_cat, 1000, norm = mpl.colors.Normalize(-wlim_cat, wlim_cat), cmap = mpl.cm.get_cmap('RdBu'))
# ax_main2.contourf(xvec, yvec, W_qsp, 1000, norm = mpl.colors.Normalize(-wlim_qsp, wlim_qsp), cmap = mpl.cm.get_cmap('RdBu'))
# ax_main2.set_title('Cat State', fontsize = 48)
ax_main2.set(xlabel = "x", ylabel = "p")

ax_xDist2.plot(xvec, marginalx_cat)
ax_xDist2.set(ylabel = 'x marginal')

ax_pDist2.plot(marginalp_cat, yvec)
ax_pDist2.set(xlabel = 'p marginal')

save_title = "20230908_cat_state_9i_corrected_title_d_9_k_1_sqrt2_size_6_res_2400_N_400_w_marginals"
fig2.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight')

plt.show()

fig3 = plt.figure(figsize = (8, 8))
gs3 = gridspec.GridSpec(3, 3)
ax_main = plt.subplot(gs3[1:3, :2])
ax_xDist = plt.subplot(gs3[0, :2], sharex = ax_main3)
ax_pDist = plt.subplot(gs3[1:3, 2],sharey = ax_main3)

ax_main.contourf(xvec, yvec, W_qsp, 1000, norm = mpl.colors.Normalize(-wlim_qsp, wlim_qsp), cmap = mpl.cm.get_cmap('RdBu'))
# ax_main.contourf(xvec, yvec, W_cat, 1000, norm = mpl.colors.Normalize(-wlim_cat, wlim_cat), cmap = mpl.cm.get_cmap('RdBu'))
# ax_main.set_title('QSPI Sensing State', fontsize = 48)
ax_main3.set(xlabel = "x", ylabel = "p")

ax_xDist3.plot(xvec, marginalx_cat)
ax_xDist3.set(ylabel = 'x marginal')

ax_pDist3.plot(marginalp_cat, yvec)
ax_pDist3.set(xlabel = 'p marginal')

save_title = "20230908_sensing_wigner_title_d_9_k_1_sqrt2_size_6_res_2400_N_400_w_marginals"
fig3.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight')

plt.show()
