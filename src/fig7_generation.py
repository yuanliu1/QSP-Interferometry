from cvqi import cvqIon
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from qutip import *
from qutip.measurement import measure
from scipy.special import factorial
import matplotlib.animation as animation
import numpy as np
from scipy.signal import find_peaks

mass = 1  # Mass of ion
q = 1  # Ion charge
N = 200  # Size of motional Hilbert space or number of Fock states
w_axial = 1  # Trap axial secular frequency
modes = 1  # Number of motional modes
psi_0 = tensor(fock(N, 0), (basis(2, 0)).unit()) # Initial state for cat state construction
ion = cvqIon(mass, q, w_axial, N, modes, psi_0) # cvqIon object to process into cat state
mode = 0

Z0, Z1 = ket2dm(basis(2, 0)), ket2dm(basis(2, 1)) # density matrices for both basis states of the qubit
PZ1 = [tensor(qeye(N), Z0), tensor(qeye(N), Z1)] # operators for the identity operator on the qubit and projectors onto both basis states of the qubit

plt.rc('font', size=42)          # controls default text sizes
plt.rc('axes', titlesize=42)     # fontsize of the axes title
plt.rc('axes', labelsize=42)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=42)    # fontsize of the tick labels
plt.rc('ytick', labelsize=42)    # fontsize of the tick labels
plt.rc('legend', fontsize=42)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title

# Phases for both the optimal QSPI state and the cat state for degrees 5, 9, and 13 with k = 0.15
k = 0.15
qspi_state_phase_list = [[3.53424467, 3.53731684, 6.28736476, 0.97696186, 3.65382249, 2.80176403, 3.4739108, 3.4831497, 5.27339552, 3.3720391, 2.25860482, 2.71654661, 3.04571734, 1.36224759], [2.29951063e+00, 5.44692069e-07, 5.25565913e+00, 3.14810245e+00, 5.73488206e+00, 6.28886041e+00, 1.10219726e+00, 8.54190371e-07, 3.68175473e+00, 8.66110002e-01], [2.3645263, 2.1434414, 3.1358502, 3.5574622, 3.1575634, 0.75503963]]
cat_state_phase_list = [[np.pi / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [np.pi / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [np.pi / 4, 0, 0, 0, 0, 0]]

cat_state_wigner_f_list = []
cat_state_wigner_g_list = []
qspi_state_wigner_f_list = []
qspi_state_wigner_g_list = []
wlim_list = []

for phases_idx in range(len(qspi_state_phase_list)):
    angles_cat = cat_state_phase_list[phases_idx]
    d = len(angles_cat)
    for i in range(len(angles_cat)):
        # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
        ion.carrier_evo(mode, 0, -2 * angles_cat[i], 1)
        ion.cntrl_disp_evo(mode, 1j * k, 1)
    
    ion.carrier_evo(mode, 0, np.pi / 2, 1)
    
    psi = ion.history_without_noise
    
    psi_0_qspi = tensor(fock(N, 0), (basis(2, 0)).unit())
    ion_qspi = cvqIon(mass, q, w_axial, N, modes, psi_0_qspi)
    angles_qspi = qspi_state_phase_list[phases_idx]
    
    d = len(angles_qspi)
    for i in range(len(angles_qspi)):
        # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
        ion_qspi.carrier_evo(mode, 0, -2 * angles_qspi[i], 1)
        ion_qspi.cntrl_disp_evo(mode, 1j * k, 1)
    
    ion_qspi.carrier_evo(mode, 0, np.pi / 2, 1)
    
    psi_qspi = ion_qspi.history_without_noise
    
    # Plot
    print("Making cat state F and G plots")
    
    xvec = np.linspace(-10, 10, 200)  # Plot x, y axes
    
    # Calculation of all relevant Wigner functions
    W1_cat = wigner((tensor(qeye(N), create(2)) * psi[-1]).ptrace(0).unit(), xvec, xvec)
    W2_cat = wigner((tensor(qeye(N), destroy(2)) * psi[-1]).ptrace(0).unit(), xvec, xvec)
    W1_qspi = wigner((tensor(qeye(N), create(2)) * psi_qspi[-1]).ptrace(0).unit(), xvec, xvec)
    W2_qspi = wigner((tensor(qeye(N), destroy(2)) * psi_qspi[-1]).ptrace(0).unit(), xvec, xvec)
    
    marginalx_cat_f = np.trapz(W1_cat, xvec, axis = 0)
    marginalp_cat_f = np.trapz(W1_cat, xvec, axis = 1)
    
    # Limits of all Wigner functions so that all of the color scales can be calibrated correctly and uniformly across the different Wigner plots
    wlim = max(abs(W1_cat).max(), abs(W2_cat).max(), abs(W1_qspi).max(), abs(W2_qspi).max())
    wlim_list.append(wlim)
    xvec = np.linspace(-10, 10, 200)  # Plot x, y axes
    
    # Saving the Wigner plot data so that the Wigner plots do not need to be recalculated
    cat_state_wigner_f_list.append(W1_cat)
    cat_state_wigner_g_list.append(W2_cat)
    qspi_state_wigner_f_list.append(W1_qspi)
    qspi_state_wigner_g_list.append(W2_qspi)

# Take the maximum across all limits of Wigner plot data so that the color map can be calibrated correctly
wlim = max(wlim_list)

# Iterate over all of the compiled data with the relevant limits now calculated in order to produce all of the necessary plots
# Please see the file cvqi_and_cat_wigner_plots.py for more complete explanations of the following code if necessary
for phases_idx in range(len(qspi_state_phase_list)):
    angles_cat = cat_state_phase_list[phases_idx]
    d = len(angles_cat)
    
    W1_cat = cat_state_wigner_f_list[phases_idx]
    W2_cat = cat_state_wigner_g_list[phases_idx]
    W1_qspi = qspi_state_wigner_f_list[phases_idx]
    W2_qspi = qspi_state_wigner_g_list[phases_idx]
    
    fig_cat_f_marg = plt.figure(figsize = (24, 20))
    gs_cat_f = gridspec.GridSpec(3, 3)
    ax_cat_f_main = plt.subplot(gs_cat_f[1:3, :2])
    ax_cat_f_xDist = plt.subplot(gs_cat_f[0, :2], sharex=ax_cat_f_main)
    ax_cat_f_pDist = plt.subplot(gs_cat_f[1:3, 2], sharey=ax_cat_f_main)
    
    W1_cat_f_marg_cb = ax_cat_f_main.contourf(xvec, xvec, W1_cat, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    ax_cat_f_main.set(xlabel="x", ylabel="p")
    
    ax_cat_f_xDist.plot(xvec, marginalx_cat_f)
    ax_cat_f_xDist.set(ylabel='x marginal')
    
    ax_cat_f_pDist.plot(marginalp_cat_f, xvec)
    ax_cat_f_pDist.set(xlabel='p marginal')
    
    #colorbar = plt.colorbar(mappable = W1_cat_f_marg_cb, ax = ax_cat_f_pDist)
    
    plt.tight_layout()
    plt.draw()
    
    save_title = "20230918_cat_state_f_d_" + str(d - 1) + "_k_15e-2_size_20_res_200_N_200_symlognorm_linthresh_0025_linscale_0025_w_marginals_colorbar.png"
    fig_cat_f_marg.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight')
    
    plt.show()
    plt.clf()
    
    print("Finished cat state F plot")
    
    marginalx_cat_g = np.trapz(W2_cat, xvec, axis = 0)
    marginalp_cat_g = np.trapz(W2_cat, xvec, axis = 1)
    
    fig_cat_g_marg = plt.figure(figsize = (24, 20))
    gs_cat_g = gridspec.GridSpec(3, 3)
    ax_cat_g_main = plt.subplot(gs_cat_g[1:3, :2])
    ax_cat_g_xDist = plt.subplot(gs_cat_g[0, :2], sharex=ax_cat_g_main)
    ax_cat_g_pDist = plt.subplot(gs_cat_g[1:3, 2], sharey=ax_cat_g_main)
    
    W2_cat_g_marg_cb = ax_cat_g_main.contourf(xvec, xvec, W2_cat, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    ax_cat_g_main.set(xlabel="x", ylabel="p")
    
    ax_cat_g_xDist.plot(xvec, marginalx_cat_g)
    ax_cat_g_xDist.set(ylabel='x marginal')
    
    ax_cat_g_pDist.plot(marginalp_cat_g, xvec)
    ax_cat_g_pDist.set(xlabel='p marginal')
    
    #plt.colorbar(mappable = W2_cat_g_marg_cb, ax = ax_cat_g_pDist)
    
    plt.tight_layout()
    plt.draw()
    
    save_title = "20230918_cat_state_g_d_" + str(d - 1) + "_k_15e-2_size_20_res_200_N_200_symlognorm_linthresh_0025_linscale_0025_w_marginals_no_colorbar.png"
    fig_cat_g_marg.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight')
    
    plt.show()
    plt.clf()
    
    print("Finished cat state G plot")
    
    
    
    # Plot
    print("Making QSPI state F and G plots")
    
    
    fig_qspi_f_marg = plt.figure(figsize = (24, 20))
    gs_qspi_f = gridspec.GridSpec(3, 3)
    ax_qspi_f_main = plt.subplot(gs_qspi_f[1:3, :2])
    ax_qspi_f_xDist = plt.subplot(gs_qspi_f[0, :2], sharex=ax_qspi_f_main)
    ax_qspi_f_pDist = plt.subplot(gs_qspi_f[1:3, 2], sharey=ax_qspi_f_main)
    
    marginalx_qspi_f = np.trapz(W1_qspi, xvec, axis = 0)
    marginalp_qspi_f = np.trapz(W1_qspi, xvec, axis = 1)
    
    W1_qspi_f_marg_cb = ax_qspi_f_main.contourf(xvec, xvec, W1_qspi, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    ax_qspi_f_main.set(xlabel="x", ylabel="p")
    
    ax_qspi_f_xDist.plot(xvec, marginalx_qspi_f)
    ax_qspi_f_xDist.set(ylabel='x marginal')
    
    ax_qspi_f_pDist.plot(marginalp_qspi_f, xvec)
    ax_qspi_f_pDist.set(xlabel='p marginal')
    
    #plt.colorbar(mappable = W1_qspi_f_marg_cb, ax = ax_qspi_f_pDist)
    
    plt.tight_layout()
    plt.draw()
    
    save_title = "20230918_qspi_state_f_d_" + str(d - 1) + "_k_15e-2_size_20_res_200_N_200_symlognorm_linthresh_0025_linscale_0025_w_marginals_no_colorbar.png"
    fig_qspi_f_marg.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight')
    
    plt.show()
    plt.clf()
    
    print("Finished QSPI state F plot")
    
    fig_qspi_g_marg = plt.figure(figsize = (24, 20))
    gs_qspi_g = gridspec.GridSpec(3, 3)
    ax_qspi_g_main = plt.subplot(gs_qspi_g[1:3, :2])
    ax_qspi_g_xDist = plt.subplot(gs_qspi_g[0, :2], sharex=ax_qspi_g_main)
    ax_qspi_g_pDist = plt.subplot(gs_qspi_g[1:3, 2], sharey=ax_qspi_g_main)
    
    marginalx_qspi_g = np.trapz(W2_qspi, xvec, axis = 0)
    marginalp_qspi_g = np.trapz(W2_qspi, xvec, axis = 1)
    
    W2_qspi_g_marg_cb = ax_qspi_g_main.contourf(xvec, xvec, W2_qspi, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    ax_qspi_g_main.set(xlabel="x", ylabel="p")
    
    ax_qspi_g_xDist.plot(xvec, marginalx_qspi_g)
    ax_qspi_g_xDist.set(ylabel='x marginal')
    
    ax_qspi_g_pDist.plot(marginalp_qspi_g, xvec)
    ax_qspi_g_pDist.set(xlabel='p marginal')
    
    #plt.colorbar(mappable = W2_qspi_g_marg_cb, ax = ax_qspi_g_pDist)
    
    plt.tight_layout()
    plt.draw()
    
    save_title = "20230918_qspi_state_g_d_" + str(d - 1) + "_k_15e-2_size_20_res_200_N_200_symlognorm_linthresh_0025_linscale_0025_w_marginals_no_colorbar.png"
    fig_qspi_g_marg.savefig(save_title, format = 'png', dpi = 1024, bbox_inches = 'tight')
    
    plt.show()
    plt.clf()
    
    print("Finished QSPI state G plot")
