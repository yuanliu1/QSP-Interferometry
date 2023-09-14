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
# Thresh = np.pi / (4 * k)
# Num angles = 40
# k = 0.1
# angles = [2.5772, 1.8177, 2.7632, 3.0494, 2.7979, 4.0244, 3.0366, 3.1596, 0.4209, 0.8344, 2.9951, 6.0169, 3.8055, 5.1502, 3.5425, 4.0559, 5.7373, 5.1349, 4.8983, 5.7997, 0.8813, 5.5616, 2.4190, 5.7434, 3.3745, 5.8709, 4.2946, 4.5547, 1.6154, 4.9453, 3.0398, 0.4331, 5.6454, 0.2510, 3.1983, 1.7248, 1.1524, 5.4260, 2.2545, 6.3369]

# Initialize k and the cat state phases for degree 9
k = 0.5
angles = [np.pi/4, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Cat state situation

d = len(angles) # set d equal to the degree of the cat state produced with the phases (i.e., the number of phases minus one)
for i in range(len(angles)): # iterate over the cat-state phases
    # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
    ion.carrier_evo(mode, 0, -2 * angles[i], 1) # perform qubit rotations
    ion.cntrl_disp_evo(mode, 1j * k, 1) # perform controlled displacements by distance k in the direction perpendicular to the signal displacement

ion.carrier_evo(mode, 0, np.pi / 2, 1) # perform final rotation to improve visualization of state in Wigner plot

psi = ion.history_without_noise # extract the final state of the qubit+oscillator system for the cat state

# Plot
print("Making plot")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 22)) # make the figure and the axes for plotting the Wigner plots of the cat state
xvec = np.linspace(-10, 10, 4000)  # Set x- and y-ranges for the Wigner plots

# Compute Wigner functions for both F and G
W1 = wigner((tensor(qeye(N), create(2)) * psi[-1]).ptrace(0).unit(), xvec, xvec) # Wigner function for F
W2 = wigner((tensor(qeye(N), destroy(2)) * psi[-1]).ptrace(0).unit(), xvec, xvec) # Wigner function for G

with open('20230912_cat_state_f_d_9_k_1_2_size_20_res_4000_N_200.npy', 'wb+') as f: # save Wigner function for F
    np.save(f, W1)
    
with open('20230912_cat_state_g_d_9_k_1_2_size_20_res_4000_N_200.npy', 'wb+') as f: # save Wigner function for G
    np.save(f, W2)

# Marginalize Wigner function for F for both quadratures of the cat state
marginalx_cat = np.trapz(W1, xvec, axis = 0)
marginalp_cat = np.trapz(W1, xvec, axis = 1)

# compute normalization for plot of F and G
wlim_cat = max(abs(W1).max(), abs(W2).max())
xvec = np.linspace(-10, 10, 4000)  # Set x- and y-ranges for plot
ax1.contourf(xvec, xvec, W1, 100, norm = mpl.colors.Normalize(-wlim_cat, wlim_cat), cmap = mpl.cm.get_cmap('RdBu')) # make contour plot of F with this normalization
ax1.set_title('QSP F')
plt.draw()

ax2.contourf(xvec, xvec, W2, 100, norm = mpl.colors.Normalize(-wlim_cat, wlim_cat), cmap = mpl.cm.get_cmap('RdBu')) # make contour plot of G with this normalization
ax2.set_title('QSP G')
plt.draw()

save_title = "20230912_cat_state_d_9_k_1_2_size_20_res_4000_N_200" # construct appropriate title for this plot
fig.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight') # save figure using this title

print("Finished plot")
plt.show()

plt.clf()

fig_cat_marg = plt.figure(figsize = (8, 8)) # create plot for showing Wigner plot of the cat state with quadrature marginals
gs_cat = gridspec.GridSpec(3, 3) # make Gridspec layout for this plot
ax_cat_main = plt.subplot(gs_cat[1:3, :2]) # make the Wigner plot area of the plot
ax_cat_xDist = plt.subplot(gs_cat[0, :2],sharex = ax_cat_main) # make the area for the x-marginal in the plot
ax_cat_pDist = plt.subplot(gs_cat[1:3, 2],sharey = ax_cat_main) # make the area for the p-marginal in the plot

ax_cat_main.contourf(xvec, xvec, W1, 100, norm = mpl.colors.Normalize(-wlim_cat, wlim_cat), cmap = mpl.cm.get_cmap('RdBu')) # make a contour plot for the Wigner plot of the cat state in the appropriate frame
#ax_main.set_title('Cat State', fontsize = 48)
ax_cat_main.set(xlabel = "x", ylabel = "p")

ax_cat_xDist.plot(xvec, marginalx_cat) # plot the x-marginal for the cat state in the appropriate frame
ax_cat_xDist.set(ylabel = 'x marginal')

ax_cat_pDist.plot(marginalp_cat, xvec) # plot the p-marginal for the cat state in the appropriate frame
ax_cat_pDist.set(xlabel = 'p marginal')

plt.draw()

save_title = "20230912_cat_state_d_9_k_1_2_size_20_res_4000_N_200_w_marginals" # construct appropriate title for this plot
fig_cat_marg.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight') # save figure using this title

plt.show()
plt.clf()

# initialize QSPI phases for degree 9
angles = [2.28907563e+00, 5.32992720e-07, 5.29348575e+00, 3.14159173e+00, 5.76908827e+00, 6.28318418e+00, 1.07601037e+00, 8.43953820e-07, 3.63649790e+00, 8.83930272e-01]


psi_0_qspi = tensor(fock(N, 0), (basis(2, 0)).unit()) # ground state for a cvqIon (qubit+oscillator system)
ion_qspi = cvqIon(mass, q, w_axial, N, modes, psi_0_qspi) # initialize cvqIon (qubit+oscillator system) in ground state

d = len(angles) # set d equal to the degree of the QSPI state produced with the phases (i.e., the number of phases minus one)
for i in range(len(angles)): # iterate over the cat-state phases
    # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
    ion_qspi.carrier_evo(mode, 0, -2 * angles[i], 1) # perform qubit rotations
    ion_qspi.cntrl_disp_evo(mode, 1j * k, 1) # perform controlled displacements by distance k in the direction perpendicular to the signal displacement

ion_qspi.carrier_evo(mode, 0, np.pi / 2, 1) # perform final rotation to improve visualization of state in Wigner plot

psi_qspi = ion_qspi.history_without_noise # extract the final state of the qubit+oscillator system for the QSPI sensing state

# Plot
print("Making plot")

fig_qspi, (ax1_qspi, ax2_qspi) = plt.subplots(2, 1, figsize = (10, 22)) # make the figure and the axes for plotting the Wigner plots of the QSPI sensing state
xvec = np.linspace(-10, 10, 4000)  # Set x- and y-ranges for the Wigner plots

# Compute Wigner functions for both F and G
W1_qspi = wigner((tensor(qeye(N), create(2)) * psi_qspi[-1]).ptrace(0).unit(), xvec, xvec) # Wigner function for F
W2_qspi = wigner((tensor(qeye(N), destroy(2)) * psi_qspi[-1]).ptrace(0).unit(), xvec, xvec) # Wigner function for G

with open('20230912_qspi_state_f_d_9_k_1_2_size_20_res_4000_N_200.npy', 'wb+') as f: # save Wigner function for F
    np.save(f, W1_qspi)
    
with open('20230912_qspi_state_g_d_9_k_1_2_size_20_res_4000_N_200.npy', 'wb+') as f: # save Wigner function for G
    np.save(f, W2_qspi)
# Marginalize Wigner function for F for both quadratures of the cat state
marginalx_qspi = np.trapz(W1_qspi, xvec, axis = 0)
marginalp_qspi = np.trapz(W1_qspi, xvec, axis = 1)

# compute normalization for plot of F and G
wlim_qspi = max(abs(W1_qspi).max(), abs(W2_qspi).max())
xvec = np.linspace(-10, 10, 4000)  # Set x- and y-ranges for plot
ax1_qspi.contourf(xvec, xvec, W1_qspi, 100, norm = mpl.colors.Normalize(-wlim_qspi, wlim_qspi), cmap = mpl.cm.get_cmap('RdBu')) # make contour plot of F with this normalization
ax1_qspi.set_title('QSP F')
plt.draw()

ax2_qspi.contourf(xvec, xvec, W2_qspi, 100, norm = mpl.colors.Normalize(-wlim_qspi, wlim_qspi), cmap = mpl.cm.get_cmap('RdBu')) # make contour plot of G with this normalization
ax2_qspi.set_title('QSP G')
plt.draw()

save_title = "20230912_qspi_state_d_9_k_1_2_size_20_res_4000_N_200" # construct appropriate title for this plot
fig_qspi.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight') # save figure using this title

print("Finished plot")
plt.show()
plt.clf()

fig_qspi_marg = plt.figure(figsize = (8, 8)) # create plot for showing Wigner plot of the QSPI sensing state with quadrature marginals
gs_qspi = gridspec.GridSpec(3, 3) # make Gridspec layout for this plot
ax_qspi_main = plt.subplot(gs_qspi[1:3, :2]) # make the Wigner plot area of the plot
ax_qspi_xDist = plt.subplot(gs_qspi[0, :2],sharex = ax_qspi_main) # make the area for the x-marginal in the plot
ax_qspi_pDist = plt.subplot(gs_qspi[1:3, 2],sharey = ax_qspi_main) # make the area for the p-marginal in the plot

ax_qspi_main.contourf(xvec, xvec, W1_qspi, 100, norm = mpl.colors.Normalize(-wlim_qspi, wlim_qspi), cmap = mpl.cm.get_cmap('RdBu')) # make a contour plot for the Wigner plot of the QSPI sensing state in the appropriate frame
#ax_main.set_title('Cat State', fontsize = 48)
ax_qspi_main.set(xlabel = "x", ylabel = "p")

ax_qspi_xDist.plot(xvec, marginalx_qspi) # plot the x-marginal for the cat state in the appropriate frame
ax_qspi_xDist.set(ylabel = 'x marginal')

ax_qspi_pDist.plot(marginalp_qspi, xvec) # plot the p-marginal for the cat state in the appropriate frame
ax_qspi_pDist.set(xlabel = 'p marginal')

plt.draw()

save_title = "20230912_qspi_state_d_9_k_1_2_size_20_res_4000_N_200_w_marginals" # construct appropriate title for this plot
fig_qspi_marg.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight') # save figure using this title

plt.show()
plt.clf()
