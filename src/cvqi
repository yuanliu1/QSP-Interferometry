import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.special import factorial
import matplotlib.animation as animation
from qutip import *


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
        H = -theta*(self.sigma_p * np.exp(-1j * phi) + self.sigma_m * np.exp(1j * phi))/2

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
        H = alpha * a.dag() - np.conj(alpha) * a

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
        H = (alpha * a.dag() - np.conj(alpha) * a) * self.sigma_z  # (np.exp(1j * phi) * self.sigma_p + np.exp(-1j * phi) * self.sigma_m)

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
