import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import numpy as np
from scipy.special import factorial
from scipy.constants import hbar
import matplotlib.animation as animation
from qutip import *
import torch
import scipy.linalg as la

def qutip_to_torch(qobj):
  return torch.tensor(qobj.data.todense(), dtype = torch.cfloat)

def qutip_to_numpy(qobj):
  return np.array(qobj.data.todense(), dtype = np.cdouble)

def torch_to_qutip_op(tensor):
  N = tensor.shape[0] // 2
  shape = [[N, 2], [N, 2]]
  return Qobj(inpt = tensor.detach().numpy(), dims = shape)

def torch_to_qutip_ket(tensor):
  N = tensor.shape[0] // 2
  shape = [[N, 2], [1, 1]]
  return Qobj(inpt = tensor.detach().numpy(), dims = shape)

def numpy_to_qutip(arr):
  N = arr.shape[0] // 2
  shape = [[N, 2], [N, 2]]
  return Qobj(inpt = arr, shape = shape)

def time_evolution_exact(state, H, t):
  #print("Hamiltonian:", H)
  result = torch_to_qutip_ket(torch.matrix_exp(-1.0j * qutip_to_torch(H) * t) @ qutip_to_torch(state))
  #print("Time evolution operator:", result)
  return result

class cvqIon:

    def __init__(self, mass, charge, trap_f, N, modes, psi):
        self.mass = mass
        self.charge = charge
        self.trap_f = trap_f
        self.N = N
        self.modes = modes

        if dims(psi)[0][-1] != 2:
            raise ValueError("Last mode should be qubit mode")

        self.psi_with_noise = psi
        self.psi_without_noise = psi
        self.times = [0]
        self.history_with_noise = [psi]
        self.history_without_noise = [psi]

        self.a_ops = []
        self.sigma_m = None
        self.sigma_z = None
        self.sigma_p = None
        self.identity = None
        self.c_ops = []
        self.h0 = None

        self.__create_a()
        self.__create_h0()

        # arbitrary noise parameters--should be taken as inputs once we determine how to model noise
        self.__add_collapse_operators(0, 0, 0.05, 0)

    def __create_a(self):
        """ Creates annihilation operators for qubit and motional modes in the order provided by initial psi
        """
        dim_list = dims(self.psi_with_noise)[0]  # List of dimensions of each mode
        ident = [qeye(dim) for dim in dim_list]  # List of operators to tensor
        for i in range(len(dim_list) - 1):  # Create annihilation operator for each mode (including qubit)
            op_list = ident[:i] + [destroy(dim_list[i])] + ident[
                                                              i + 1:]  # Create operator list for annihilation of each motional mode
            self.a_ops.append(tensor(op_list))  # Create annihilation operator for motional mode
        self.sigma_m = tensor(ident[:-1] + [destroy(dim_list[-1])])  # Construct sigma_m operator
        self.sigma_z = tensor(ident[:-1] + [sigmaz()])  # Construct sigma_z operator
        self.sigma_p = self.sigma_m.dag()
        self.identity = tensor(ident[:-1] + [qeye(dim_list[-1])])

    def __create_h0(self):
        """Creates free evolution / base hamiltonian
        """
        # Simple QHO at secular trap freqeuncy
        # todo: add higher order terms of trap potential
        total_h0 = 0
        for mode in range(0, self.modes):
            a = self.a_ops[mode]
            total_h0 += self.trap_f *(a * a.dag()-self.identity)
        self.h0 = total_h0

    def __add_collapse_operators(self, spontaneous_emission_rate, dephasing_rate, motional_heating_rate,
                                 motional_decay_rate):
        """Collapse operators for the master equation of a single atom and a harmonic oscillator
        Support ONE motional mode currently

        returns: list of collapse operators for master equation solution of atom + harmonic oscillator
        """
        rate = spontaneous_emission_rate
        if rate > 0.0:
            self.c_ops.append(np.sqrt(rate) * self.sigma_m)

        rate = dephasing_rate
        if rate > 0.0:
            self.c_ops.append(np.sqrt(rate) * self.sigma_z)
            rate = dephasing_rate

        rate = motional_decay_rate
        if rate > 0.0:
            self.c_ops.append(np.sqrt(rate) * self.a_ops[0])

        rate = motional_heating_rate
        if rate > 0.0:
            self.c_ops.append(np.sqrt(rate) * self.a_ops[0].dag())

    def lamb_dicke(self, wavelength, theta):
        """computes the Lamb Dicke parameter

        @ var wavelength: laser wavelength in meters
        @ var theta: laser projection angle in degrees

        returns: Lamb-Dicke parameter
        """
        k = 2. * np.pi / wavelength
        return k * np.sqrt(1 / (2 * self.mass * 2 * np.pi * self.trap_f)) * np.abs(np.cos(np.pi * theta / 180))

    # Following are CVQC operations to perform on ion object
    # Currently using interaction picture and mesolve
    # self.psi can be state vector or density matrix
    # todo: change alpha/sq/chi etc into experimental parameters

    def __apply_hamiltonian(self, H, tlist, noise=False, solver='me', e_ops=[], a_ops=[], args={}):
        """ Applies the given Hamiltonian over the given set of times with the given solver

        @ var H: the Hamiltonian under which the principal system is evolving
        @ var tlist: total time; input as float or list of floats
        @ mode: the solver to be used for the numerical solution
        @ e_ops: the expectation values to be calculated for each timestep
        @ args: any arguments to the time-dependent Hamiltonian terms

        returns present state of principal system with noisy evolution
        """
        # Construct an appropriate array of times
        if type(tlist) is not np.ndarray:
            tlist = np.linspace(0, tlist, 500)

        # Select the appropriate function for solving
        if solver == 'brme':
            solver_func = brmesolve
        elif solver == 'fmme':
            solver_func = fmmesolve
        else:
            solver_func = mesolve

        # Solve both with noise and without
        result_without_noise = mesolve(H, self.psi_without_noise, tlist, c_ops=[], e_ops=e_ops, args=args)
        result_with_noise = result_without_noise
        if noise:
            result_with_noise = solver_func(H, self.psi_with_noise, tlist, c_ops=self.c_ops, e_ops=e_ops, args=args)

        # Store the results and return the final result with noise
        if e_ops == []:
            self.psi_with_noise = result_with_noise.states[-1]
            self.psi_without_noise = result_without_noise.states[-1]
            self.times.extend([self.times[-1] + t for t in result_with_noise.times])
            self.history_with_noise.extend(result_with_noise.states)
            self.history_without_noise.extend(result_without_noise.states)

        return result_with_noise, result_without_noise

    def rsb_evo(self, mode: int, n: int, eta: float, phi: float, tlist: float, e_ops=[], solver='me'):
        """ Solves time evolution for n'th red sideband transition

        @ var mode: Mode of ion to perform operation on
        @ var n: n'th order of sideband
        @ var eta: Lamb-Dicke parameter
        @ var phi: Phase of laser
        @ var tlist: Total time; input as float or list of floats
        @ var e_ops: Expectation values to calculate
        @ var solver: Choose which QuTip solver to use (default mesolve())

        returns result: list of evolved states at each time in tlist
        """
        a = self.a_ops[mode]
        # Dipole coupling <g|E_0 * x|e>
        omegaR = 1  # What is OmegaR for our ion?
        # First RSB hamiltonian
        H = 1 / 2 * omegaR * eta ** n / factorial(n) * (
                    a ** n * self.sigma_p * np.exp(1j * phi) + a.dag() ** n * self.sigma_m * np.exp(-1j * phi))

        return self.__apply_hamiltonian(H, tlist, solver=solver, e_ops=e_ops)

    def bsb_evo(self, mode: int, n: int, eta: float, phi: float, tlist: float, e_ops=[], solver='me'):
        """ Solves time evolution for n'th blue sideband transition

        @ var mode: Mode of ion to perform operation on
        @ var n: n'th order of sideband
        @ var eta: Lamb-Dicke parameter
        @ var phi: Phase of laser
        @ var tlist: Total time; input as float or list of floats
        @ var e_ops: Expectation values to calculate
        @ var solver: Choose which QuTip solver to use (default mesolve())

        returns result: list of evolved states at each time in tlist
        """
        a = self.a_ops[mode]
        # Dipole coupling <g|E_0 * x|e>
        omegaR = 1
        H = 1 / 2 * omegaR * eta ** n / factorial(n) * (
                    a.dag() ** n * self.sigma_p * np.exp(1j * phi) + a ** n * self.sigma_m * np.exp(-1j * phi))

        return self.__apply_hamiltonian(H, tlist, solver=solver, e_ops=e_ops)

    def carrier_evo(self, mode: int, phi: float, theta: float, tlist: float, e_ops=[], solver='me'):
        """ Solves time evolution for n'th blue sideband transition

        @ var mode: Mode of ion to perform operation on
        @ var phi: Phase of laser
        @ var tlist: Total time; input as float or list of floats
        @ var e_ops: Expectation values to calculate
        @ var solver: Choose which QuTip solver to use (default mesolve())

        returns result: list of evolved states at each time in tlist
        """
        a = self.a_ops[mode]
        # Dipole coupling <g|E_0 * x|e>
        # omegaR = 1
        H = -theta * (self.sigma_p * np.exp(-1j * phi) + self.sigma_m * np.exp(1j * phi))/2

        return self.__apply_hamiltonian(H, tlist)

    def free_evo(self, mode: int, tlist: float, e_ops=[], solver='me'):
        """ Solves time evolution of psi under free hamiltonian

        @ var tlist: total time; input as float or list of floats

        returns result: list of evolved states at each time in tlist
        """
        return self.__apply_hamiltonian(self.h0, tlist, solver=solver, e_ops=e_ops)

    def disp_evo(self, mode: int, alpha: complex, tlist: float):
        """ Solves time evolution of psi under displacement hamiltonian

        @ var mode: Mode of ion to perform operation
        @ var alpha: Complex displacement parameter
        @ var tlist: Total time; input as float or list of floats

        returns result: list of evolved states at each time in tlist
        """

        a = self.a_ops[mode]
        H = 1j * (alpha * a.dag() - np.conj(alpha) * a)
        #print("Hamiltonian:", H)

        return self.__apply_hamiltonian(H, tlist)

    def sq_evo(self, mode: int, sq: complex, tlist: float):
        """ Solves time evolution of psi under squeezing hamiltonian

        @ var mode: Mode of ion to perform operation
        @ var sq: Complex squeeze factor
        @ var tlist: Total time; input as float or list of floats

        returns result: list of evolved states at each time in tlist
        """

        a = self.a_ops[mode]
        H = 1 / 2 * (np.conj(sq) * a * a - sq * a.dag() * a.dag())

        return self.__apply_hamiltonian(H, tlist)

    def phase_evo(self, mode: int, phi: float, tlist: float):
        """ Solves time evolution of psi under phase shift hamiltonian

        @ var mode: Mode of ion to perform operation
        @ var phi: Phase
        @ var tlist: Total time; input as float or list of floats

        returns result: list of evolved states at each time in tlist
        """

        a = self.a_ops[mode]
        H = -1j * phi * a.dag() * a

        return self.__apply_hamiltonian(H, tlist)

    def kerr_evo(self, mode: int, chi: float, tlist: float):
        """ Solves time evolution of psi under cross-Kerr hamiltonian

        @ var mode: Mode of ion to perform operation
        @ var chi: Kerr nonlinearity
        @ var tlist: total time; input as float or list of floats

        returns result: list of evolved states at each time in tlist
        """

        a = self.a_ops[mode]
        H = 1 / 2 * chi * a.dag() * a.dag() * a * a

        return self.__apply_hamiltonian(H, tlist)

    def bs_evo(self, mode1: int, mode2: int, theta: float, tlist: float):
        """Solves time evolution of psi under beamsplitter hamiltonian

        @ var mode1, mode2: Modes of ion to perform operation
        @ var theta: transmissivity of beam splitter
        @ var tlist: total time

        returns result: list of evolved states at each time in tlist
        """

        a = self.a_ops[mode1]
        b = self.a_ops[mode2]
        H = theta * (a.dag() * b - a * b.dag())

        return self.__apply_hamiltonian(H, tlist)

    def stark_evo(self, mode: int, tlist: float):
        """Solves time evolution of psi under AC Stark hamiltonian

        @ var mode: Mode of ion to perform operation
        @ var tlist: total time

        returns result: list of evolved states at each time in tlist
        """
        a = self.a_ops[mode]
        H = a.dag() * a * self.sigma_z

        return self.__apply_hamiltonian(H, tlist)

    def cntrl_disp_evo(self, mode: int, alpha: complex, tlist: float, phi=0):
        """Solves time evolution of psi under Control Displacement hamiltonian

        @ var mode: Mode of ion to perform operation
        @ var alpha: Complex displacement parameter
        @ var phi: Spin pauli phase
        @ var tlist: total time

        returns result: list of evolved states at each time in tlist
        """

        a = self.a_ops[mode]
        H = 1j * (alpha * a.dag() - np.conj(alpha) * a) * self.sigma_z  # (np.exp(1j * phi) * self.sigma_p + np.exp(-1j * phi) * self.sigma_m)

        return self.__apply_hamiltonian(H, tlist)

    def el_drive(self, noise: bool, noise_const: float, amp: float, phase: float, w: float, mode: int, tlist: float,
                 blackman=False, ramp_up=0, ramp_down=0, e_ops=[], solver='me'):
        """ Solves time evolution of psi under time-dependent classical electric drive field

        @ var amp: Amplitude of E-field
        @ var phase: Phase of E-field
        @ var w: E-field drive frequency
        @ var envelope: Amplitude modulation of E-field
        @ var tlist: total time; input as float or list of floats
        @ var solver: Choose which QuTip solver to use (default mesolve())
        @ var blackman: boolean specifying whether to use a Blackman window or a linear ramp-up/ramp-down
        @ var ramp_up: proportion of duration spent ramping up (must be non-negative and sum with ramp_down to at most 1 if blackman is False)
        @ var ramp_down: proportion of duration spent ramping down (must be non-negative and sum with ramp_up to at most 1 if blackman is False)
        @ var e_ops: Expectation values to calculate

        returns result: list of evolved states (or expectation values) at each time in tlist
        """

        # Find the duration (for the purpose of generating the ramp)
        T = np.max(tlist)

        a = self.a_ops[mode]

        a_ops = []

        # Rabi frequency due to an E-field drive near the trap secular frequency
        OmegaR = self.charge * amp * np.sqrt(1 / (8 * self.mass * self.trap_f))
        # OmegaR = 0.1
        d = w - self.trap_f
        s = w + self.trap_f

        h_d = OmegaR * a.dag()
        h_ddag = OmegaR * a
        h_s = OmegaR * a.dag()
        h_sdag = OmegaR * a

        h_dt = 'exp(-1j * (delta*t+phi))'
        h_ddagt = 'exp(1j * (delta*t+phi))'
        h_st = 'exp(-1j * (sigma*t+phi))'
        h_sdagt = 'exp(1j * (sigma*t+phi))'

        # Time-dependent Hamiltonian (without and with RWA)
        H = [[h_d, h_dt], [h_ddag, h_ddagt], [h_s, h_st], [h_sdag, h_sdagt]]
        H_rwa = [[h_d, h_dt], [h_ddag, h_ddagt]]
        args = {'delta': d, 'sigma': s, 'phi': phase, 'ramp_up': ramp_up, 'ramp_down': ramp_down, 'T': T}

        # Generate the ramping coefficient and prepend to each time-dependent term
        if not blackman:
            if ramp_up < 0 or ramp_down < 0 or ramp_up + ramp_down > 1:
                raise ValueError("Ramp-up and ramp-down must both be non-negative and sum to at most 1")

            rising_str = '(t <= ramp_up * T) * t / (ramp_up * T)'
            flat_str = '((t > ramp_up * T) & (t < (1 - ramp_down) * T))'
            falling_str = '(t >= (1 - ramp_down * T)) * (1 - (t - (1 - ramp_down) * T)) / (ramp_down * T)'
            ramp_str = '(' + rising_str + ' + ' + flat_str + ' + ' + falling_str + ') * '
            if ramp_up == 0 and ramp_down == 0:
                ramp_str = ''
        else:
            const_str = '7938 / 18608'
            cos2_str = '9240 / 18608 * cos(2 * pi * t / T)'
            cos4_str = '1430 / 18608 * cos(4 * pi * t / T)'
            ramp_str = '(' + const_str + ' - ' + cos2_str + ' + ' + cos4_str + ') * '

        # H_ramped = [[h, ramp_str + h_t] for [h, h_t] in H]
        H_ramped = [[h, ramp_str + h_t] for [h, h_t] in H_rwa]

        if noise:
            kappa = noise_const  # White noise constant
            a_ops = [[(a, a.dag()), ('{0}*(w >= 0)'.format(kappa), 'exp(1j*t)', 'exp(-1j*t)')]]  # White noise bath

        return self.__apply_hamiltonian(H_ramped, tlist, solver=solver, e_ops=e_ops, a_ops=a_ops, args=args)

    # Following are plotting function
    # todo: plot other functions, make robust to handling many motional modes

    def create_wigner_plot(self, psi, xvec, yvec, state = 0):
        """ Creates new Wigner plot figure of initial state

        @ var psi: list of states to plot (from mesolve)
        @ var xvec, yvec: List of x/y-coordinates to evaluate Wigner function at

        returns fig, ax: figure and axes created
        """
        fig, ax = plt.subplots()
        # W = wigner(psi.states[0].ptrace(0), xvec, xvec)  # Note: here we trace out qubit before plotting
        W = wigner(psi[state].ptrace(0), xvec, xvec)  # Note: here we trace out qubit before plotting
        wlim = abs(W).max()
        ax.contourf(xvec, yvec, W, 100, norm=mpl.colors.Normalize(-wlim, wlim), cmap=mpl.cm.get_cmap('RdBu'))
        return fig, ax

    def cont_wigner_plot(self, psi, fig, ax, xvec, yvec, frames, step, title=''):
        """Make animation of Wigner plot time evolution

        @ var psi: list of state of plot (from mesolve)
        @ var xvec, yvec: List of x/y-coordinates to evaluate Wigner function at
        @ vars fig, ax: Figure and axes to plot on (generated by create_plot())
        @ var title: add optional title to figure
        """
        ax.set_title(title)
        for n in range(0, frames - 1, step):
            plt.pause(.01)
            # W = wigner(psi.states[n].ptrace(0), xvec, yvec)
            W = wigner(psi[n].ptrace(0), xvec, yvec)
            wlim = abs(W).max()
            ax.contourf(xvec, yvec, W, 100, norm=mpl.colors.Normalize(-wlim, wlim), cmap=mpl.cm.get_cmap('RdBu'))
            plt.draw()

    def fidelity(self):
        """ Calculates the fidelity of the current state with noisy evolution relative to the alternative without noise

        returns the desired fidelity
        """
        return fidelity(self.psi_with_noise.ptrace(0), self.psi_without_noise.ptrace(0))

    def evolution_exact(self, H, t):
      state = self.psi_without_noise
      result_without_noise = time_evolution_exact(state, H, t)
      self.psi_without_noise = result_without_noise
      self.history_without_noise.extend([result_without_noise])

    def carrier_evo_exact(self, mode: int, phi: float, theta: float, t: float):
      """ Solves exact time evolution for n'th blue sideband transition

      @ var mode: Mode of ion to perform operation on
      @ var phi: Phase of laser
      @ var tlist: Total time; input as float or list of floats
      @ var e_ops: Expectation values to calculate
      @ var solver: Choose which QuTip solver to use (default mesolve())

      returns result: list of evolved states at each time in tlist
      """
      a = self.a_ops[mode]
      # Dipole coupling <g|E_0 * x|e>
      # omegaR = 1
      H = -theta * (self.sigma_p * np.exp(-1j * phi) + self.sigma_m * np.exp(1j * phi))/2

      return self.evolution_exact(H, t)

    def cntrl_disp_evo_exact(self, mode: int, alpha: complex, t: float, phi=0):
        """Solves exact time evolution of psi under Control Displacement hamiltonian

        @ var mode: Mode of ion to perform operation
        @ var alpha: Complex displacement parameter
        @ var phi: Spin pauli phase
        @ var tlist: total time

        returns result: list of evolved states at each time in tlist
        """

        a = self.a_ops[mode]
        H = 1j * (alpha * a.dag() - np.conj(alpha) * a) * self.sigma_z  # (np.exp(1j * phi) * self.sigma_p + np.exp(-1j * phi) * self.sigma_m)

        return self.evolution_exact(H, t)

    def disp_evo_exact(self, mode: int, alpha: complex, t: float):
      """ Solves exact time evolution of psi under displacement hamiltonian

      @ var mode: Mode of ion to perform operation
      @ var alpha: Complex displacement parameter
      @ var tlist: Total time; input as float or list of floats

      returns result: list of evolved states at each time in tlist
      """

      a = self.a_ops[mode]
      H = 1j * (alpha * a.dag() - np.conj(alpha) * a)

      return self.evolution_exact(H, t)

mass = 1  # Mass of ion
q = 1  # Ion charge
N = 200  # Size of motional Hilbert space or number of Fock states
w_axial = 1  # Trap axial secular frequency
modes = 1  # Number of motional modes
psi_0 = tensor(fock(N, 0), (basis(2, 0)).unit()) # Initial state for cat state construction
ion = cvqIon(mass, q, w_axial, N, modes, psi_0) # cvqIon object to process into cat state
mode = 0

LARGE_SIZE = 20

plt.rc('font', size = LARGE_SIZE)          # controls default text sizes
plt.rc('axes', titlesize = LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize = LARGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize = LARGE_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize = LARGE_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize = LARGE_SIZE)    # legend fontsize
plt.rc('figure', titlesize = LARGE_SIZE)  # fontsize of the figure title

qspi_state_phase_list = [[3.51520638e+00, 3.49773531e+00, 2.38828282e-03, 9.99732600e-01, 3.66922981e+00, 2.78454440e+00, 3.58316684e+00, 3.38466538e+00, 5.29877696e+00, 3.41527760e+00, 2.16847087e+00, 2.78526302e+00, 3.00039577e+00, 5.67806652e+00], [2.29287833e+00, 5.27992043e-07, 5.23464400e+00, 3.16089739e+00, 5.75171624e+00, 1.42697928e-02, 1.12519123e+00, 8.10798972e-07, 3.66603384e+00, 8.82856109e-01], [2.35987264, 2.12652193, 3.14787306, 3.57758637, 3.15854806, 0.77501569]]
cat_state_phase_list = [[np.pi / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [np.pi / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0], [np.pi / 4, 0, 0, 0, 0, 0]]

k = 0.15 * np.sqrt(2)

xvec = np.linspace(-10, 10, 400)
yvec = np.linspace(-10, 10, 400)

cat_state_wigner_f_list = []
cat_state_wigner_g_list = []
qspi_state_wigner_f_list = []
qspi_state_wigner_g_list = []
qspi_state_full_1_wigner_f_list = []
qspi_state_full_2_wigner_f_list = []
wlim_list = []

for phases_idx in range(len(qspi_state_phase_list)):
  # Prepare cat state
  psi_0 = tensor(fock(N, 0), (basis(2, 0)).unit()) # Initial state for cat state construction
  ion = cvqIon(mass, q, w_axial, N, modes, psi_0) # cvqIon object to process into cat state
  mode = 0
  print("Constructing cat states")
  angles_cat = cat_state_phase_list[phases_idx]
  d = len(angles_cat)
  for i in range(len(angles_cat) - 1):
      # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
      ion.carrier_evo_exact(mode, 0, -2 * angles_cat[i], 1)
      ion.cntrl_disp_evo_exact(mode, 1j * k, 1)

  # This last qubit rotation is cancelled by the reverse qubit rotation in the inverse QSP sequence, so here, we choose it to accentuate the oscillator-state features
  ion.carrier_evo_exact(mode, 0, np.pi / 2, 1)

  psi = ion.history_without_noise


  print("Finished constructing cat states and now constructing QSPI states")

  # Prepare QSPI state
  psi_0_qspi = tensor(fock(N, 0), (basis(2, 0)).unit())
  ion_qspi = cvqIon(mass, q, w_axial, N, modes, psi_0_qspi)
  angles_qspi = qspi_state_phase_list[phases_idx]

  d = len(angles_qspi)
  for i in range(len(angles_qspi) - 1):
      # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
      ion_qspi.carrier_evo_exact(mode, 0, -2 * angles_qspi[i], 1)
      ion_qspi.cntrl_disp_evo_exact(mode, 1j * k, 1)

  # The last qubit rotation is cancelled by the reverse qubit rotation in the inverse QSP sequence, so here, we choose it to accentuate the oscillator-state features
  ion_qspi.carrier_evo_exact(mode, 0, 3 * np.pi / 4, 1)

  psi_qspi = ion_qspi.history_without_noise

  print("Finished constructing QSPI states and now simulating entire protocol")

  # Prepare two states carrying out the entire protocol with two different displacements
  psi_0_qspi_full_1 = tensor(fock(N, 0), (basis(2, 0)).unit())
  psi_0_qspi_full_2 = tensor(fock(N, 0), (basis(2, 0)).unit())
  ion_qspi_full_1 = cvqIon(mass, q, w_axial, N, modes, psi_0_qspi_full_1)
  ion_qspi_full_2 = cvqIon(mass, q, w_axial, N, modes, psi_0_qspi_full_2)

  # Preparation of QSPI state by QSP
  for i in range(len(angles_qspi) - 1):
    # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
    ion_qspi_full_1.carrier_evo_exact(mode, 0, -2 * angles_qspi[i], 1)
    ion_qspi_full_2.carrier_evo_exact(mode, 0, -2 * angles_qspi[i], 1)
    ion_qspi_full_1.cntrl_disp_evo_exact(mode, 1j * k / np.sqrt(2), 1)
    ion_qspi_full_2.cntrl_disp_evo_exact(mode, 1j * k / np.sqrt(2), 1)
  ion_qspi_full_1.carrier_evo_exact(mode, 0, -2 * angles_qspi[-1], 1)
  ion_qspi_full_2.carrier_evo_exact(mode, 0, -2 * angles_qspi[-1], 1)

  # Displacement to be sensed
  ion_qspi_full_1.disp_evo_exact(mode, (0.125 * np.pi / k) / np.sqrt(2), 1)
  ion_qspi_full_2.disp_evo_exact(mode, (0.375 * np.pi / k) / np.sqrt(2), 1)

  # Undoing of QSPI preparation by inverse QSP
  for i in range(len(angles_qspi) - 1, 0, -1):
      # There is a factor of 2 because of the way we define qubit rotations in cvqi.py
      ion_qspi_full_1.carrier_evo_exact(mode, 0, 2 * angles_qspi[i], 1)
      ion_qspi_full_2.carrier_evo_exact(mode, 0, 2 * angles_qspi[i], 1)
      ion_qspi_full_1.cntrl_disp_evo_exact(mode, -1j * k / np.sqrt(2), 1)
      ion_qspi_full_2.cntrl_disp_evo_exact(mode, -1j * k / np.sqrt(2), 1)
  ion_qspi_full_1.carrier_evo_exact(mode, 0, 2 * angles_qspi[0], 1)
  ion_qspi_full_2.carrier_evo_exact(mode, 0, 2 * angles_qspi[0], 1)

  psi_qspi_full_1 = ion_qspi_full_1.history_without_noise
  psi_qspi_full_2 = ion_qspi_full_2.history_without_noise


  print("Finished simulating entire protocol")

  print("Finished constructing all states and now computing Wigner plots")

  xvec = np.linspace(-10, 10, 400)  # Plot x, y axes

  # Calculation of all relevant Wigner functions
  W1_cat = wigner((tensor(qeye(N), create(2)) * psi[-1]).ptrace(0), xvec, xvec)
  W2_cat = wigner((tensor(qeye(N), destroy(2)) * psi[-1]).ptrace(0), xvec, xvec)
  W1_qspi = wigner((tensor(qeye(N), create(2)) * psi_qspi[-1]).ptrace(0), xvec, xvec)
  W2_qspi = wigner((tensor(qeye(N), destroy(2)) * psi_qspi[-1]).ptrace(0), xvec, xvec)

  W1_qspi_full_1 = wigner((tensor(qeye(N), create(2)) * psi_qspi_full_1[-1]).ptrace(0), xvec, xvec)
  W2_qspi_full_1 = wigner((tensor(qeye(N), destroy(2)) * psi_qspi_full_1[-1]).ptrace(0), xvec, xvec)
  W1_qspi_full_2 = wigner((tensor(qeye(N), create(2)) * psi_qspi_full_2[-1]).ptrace(0), xvec, xvec)
  W2_qspi_full_2 = wigner((tensor(qeye(N), destroy(2)) * psi_qspi_full_2[-1]).ptrace(0), xvec, xvec)

  print("Probability of measuring 0 after displacement by 1/2 * beta_th:", np.sum(W1_qspi_full_1) / 400)
  print("Probability of measuring 1 after displacement by 1/2 * beta_th:", np.sum(W2_qspi_full_1) / 400)

  print("Probability of measuring 0 after displacement by 3/2 * beta_th:", np.sum(W1_qspi_full_2) / 400)
  print("Probability of measuring 1 after displacement by 3/2 * beta_th:", np.sum(W2_qspi_full_2) / 400)


  # Limits of all Wigner functions so that all of the color scales can be calibrated correctly and uniformly across the different Wigner plots
  wlim = max(abs(W1_cat).max(), abs(W2_cat).max(), abs(W1_qspi).max(), abs(W2_qspi).max()) #, abs(W1_qspi_full_1).max(), abs(W1_qspi_full_2).max())
  wlim_list.append(wlim)

  # Saving the Wigner plot data so that the Wigner plots do not need to be recalculated
  cat_state_wigner_f_list.append(W1_cat)
  cat_state_wigner_g_list.append(W2_cat)
  qspi_state_wigner_f_list.append(W1_qspi)
  qspi_state_wigner_g_list.append(W2_qspi)
  qspi_state_full_1_wigner_f_list.append(W1_qspi_full_1)
  qspi_state_full_2_wigner_f_list.append(W1_qspi_full_2)

# Take the maximum across all limits of Wigner plot data so that the color map can be calibrated correctly
wlim = max(wlim_list)
print(wlim)

# Iterate over all of the compiled data with the relevant limits now calculated in order to produce all of the necessary plots
# Please see the file cvqi_and_cat_wigner_plots.py for more complete explanations of the following code if necessary
for phases_idx in range(len(qspi_state_phase_list)):
    angles_cat = cat_state_phase_list[phases_idx]
    d = len(angles_cat)

    W1_cat = cat_state_wigner_f_list[phases_idx]
    W2_cat = cat_state_wigner_g_list[phases_idx]
    W1_qspi = qspi_state_wigner_f_list[phases_idx]
    W2_qspi = qspi_state_wigner_g_list[phases_idx]
    W1_qspi_full_1 = qspi_state_full_1_wigner_f_list[phases_idx]
    W1_qspi_full_2 = qspi_state_full_2_wigner_f_list[phases_idx]

    # Plot
    print("Making cat state F and G plots")

    plt_cat_f = plt.contourf(xvec, xvec, W1_cat, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    plt.xlabel("$x$")
    plt.ylabel("$p$")

    plt.tight_layout()
    plt.draw()

    save_title = "20231118_cat_state_f_d_" + str(d - 1) + "_k_15e-2sqrt2_size_20_res_400_N_200_symlognorm_linthresh_0025_linscale_0025_colorbar.png"
    plt.savefig(save_title, format = 'png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.clf()

    print("Finished cat state F plot")

    plt_cat_g = plt.contourf(xvec, xvec, W2_cat, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    plt.xlabel("$x$")
    plt.ylabel("$p$")

    plt.tight_layout()
    plt.draw()

    save_title = "20231118_cat_state_g_d_" + str(d - 1) + "_k_15e-2sqrt2_size_20_res_400_N_200_symlognorm_linthresh_0025_linscale_0025_no_colorbar.png"
    plt.savefig(save_title, format = 'png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.clf()

    print("Finished cat state G plot")

    # Plot
    print("Making QSPI state F and G plots")


    plt_qspi_f = plt.contourf(xvec, xvec, W1_qspi, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    plt.xlabel("$x$")
    plt.ylabel("$p$")

    plt.axis('square')
    plt.tight_layout()
    plt.draw()

    save_title = "20231118_qspi_state_f_d_" + str(d - 1) + "_k_15e-2sqrt2_size_20_res_400_N_200_symlognorm_linthresh_0025_linscale_0025_no_colorbar.png"
    plt.savefig(save_title, format = 'png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.clf()

    print("Finished QSPI state F plot")

    plt_qspi_g = plt.contourf(xvec, xvec, W2_qspi, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    plt.xlabel("$x$")
    plt.ylabel("$p$")

    plt.axis('square')
    plt.tight_layout()
    plt.draw()

    save_title = "20231118_qspi_state_g_d_" + str(d - 1) + "_k_15e-2sqrt2_size_20_res_400_N_200_symlognorm_linthresh_0025_linscale_0025_no_colorbar.png"
    plt.savefig(save_title, format = 'png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.clf()

    print("Finished QSPI state G plot")

    # Plot
    print("Making full QSPI protocol qubit 0 plots")

    plt_cat_g = plt.contourf(xvec, xvec, W1_qspi_full_1, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    plt.xlabel("$x$")
    plt.ylabel("$p$")

    plt.axis('square')
    plt.tight_layout()
    plt.draw()

    save_title = "20231118_qspi_state_qubit0_1_2_beta_th_d_" + str(d - 1) + "_k_15e-2sqrt2_size_20_res_400_N_200_symlognorm_linthresh_0025_linscale_0025_no_colorbar.png"
    plt.savefig(save_title, format = 'png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.clf()

    print("Finished QSPI state full qubit 0 plot for displacement 1/2 beta_th")

    plt_cat_g = plt.contourf(xvec, xvec, W1_qspi_full_2, 100, norm=mpl.colors.SymLogNorm(linthresh = 0.0025, linscale = 0.0025, vmin = -wlim, vmax = wlim, base = 10), cmap=mpl.cm.get_cmap('RdBu'))
    plt.xlabel("$x$")
    plt.ylabel("$p$")


    plt.axis('square')
    plt.tight_layout()
    plt.draw()

    save_title = "20231118_qspi_state_qubit0_3_2_beta_th_d_" + str(d - 1) + "_k_15e-2sqrt2_size_20_res_400_N_200_symlognorm_linthresh_0025_linscale_0025_no_colorbar.png"
    plt.savefig(save_title, format = 'png', dpi = 300, bbox_inches = 'tight')

    plt.show()
    plt.clf()

    print("Finished QSPI state full qubit 0 plot for displacement 3/2 beta_th")
