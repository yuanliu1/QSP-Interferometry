# Quantum Signal Processing Interferometry

## Introduction

[Quantum signal processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) is a framework for quantum algorithms including Hamiltonian simulation, quantum linear system solving, amplitude amplification, etc.

In this work, we make use of a generalization of standard qubit QSP to the case of an ancilla qubit coupled to a bosonic oscillator to perform signal processing of the bosonic quadrature operators.  We build on top of such **bosonic QSP** to construct a bosonic **QSP Interferometry** protocol and utilize it to perform quantum sensing of displacements on the bosonic qumode.  In particular, for our unitary U, we utilize the displacement gate with displacement $k$, so our QSP circuit is characterized by controlled displacement operations on the bosonic mode with the ancilla sensing qubit taken as the control, denoted by $c\mathcal{D}(\cdot)$, alternated with qubit rotations on the ancilla sensing qubit.

Such a circuit allows for more efficient quantum-sensing decisions because the information regarding the displacement of the state is transferred to the qubit, meaning that the outcome of the sensing experiment, and hence the binary decision, can read from the ancilla qubit with a single binary measurement.

## Using the code in this repository

This repository includes four code files, namely cvqi.py, qspi\_phase\_learning.py, cvqi\_and\_cat\_wigner\_plots.py, and plot\_qspi\_sensing\_state\_response.py.  Each code file serves a unique but important role for this work and its visualizations, so we will briefly describe each in a separate section of this README.

Note that in order to utilize this code, there are a few basic requirements regarding necessary packages.  These are outlined in the document src/requirements.txt of this repository.  Note that to install these required packages, it is necessary only to navigate to the directory in which the text file requirements.txt is located and then enter the line `pip install -r requirements.txt` in the terminal.  This will install all of the required versions of the packages for running the code for QSPI phase learning and visualization.

### cvqi.py

The cvqi.py code file provides code for simulating continuous-variable quantum computing, or CVQI, operations, on a qumode, as well as code for hybrid operations with an ancilla qubit.  The functions are outlined in the table below.

| Function Name | Description | Input Parameters |
|-|-|-|
| \_\_apply\_h | Evolution under any Hamiltonian | Hamiltonian $H$<br>Operation time $t$<br>Master Equation solver name<br>Additional arguments dict. |
| free\_evo | Free Hamiltonian evolution | Operation time $t$ |
| disp\_evo | Displacement Hamiltonian evolution | Mode number $j$<br>Displacement parameter $\alpha$<br>Operation time $t$ |
| sq\_evo | Squeezing Hamiltonian evolution | Mode number $j$<br>Parametric amp. strength $g$<br>Modulation drive phase $\theta$<br>Operation time $t$ |
| bs\_evo | Beamsplitter Hamiltonian evolution | First mode number $j$<br>Second mode number $k$<br>Beamsplitter strength $g$<br>Operation time $t$ |
| el\_drive | Time-dependent classical electric drive field evolution | Amplitude of E-field $A$<br> Phase of E-field $\phi$<br>E-field drive frequency $f$<br>Mode number $j$<br>Amplitude modulation of E-field $e$<br>Operation time $t$<br>QuTiP Solver *solver*<br>Boolean for use of Blackman window vs. linear ramp-up and ramp-down<br>Ramp-up proportion *ramp\_up*<br>Ramp-down proportion *ramp\_down* |
| phase\_evo | Phase-shift Hamiltonian evolution | Mode number $j$<br>Phase $\phi$<br>Operation time $t$ |
| rsb\_evo | Red side-band Hamiltonian evolution | Mode number $j$<br>Side-band order $n$<br>Phase $\phi$<br>Operation time $t$ |
| bsb\_evo | Blue side-band Hamiltonian evolution | Mode number $j$<br>Side-band order $n$<br>Phase $\phi$<br>Operation time $t$ |
| stark\_evo | AC Stark Hamiltonian evolution | Mode number $j$<br>Operation time $t$ |
| kerr\_evo | Cross-Kerr Hamiltonian evolution | Mode number $j$<br>Kerr nonlinearity $\chi$<br>Operation time $t$ |
| cntrl\_disp\_evo | Controlled displacement Hamiltonian evolution | Mode number $j$<br>Displacement parameter $\alpha$<br>Spin pauli phase $\phi$<br>Operation time $t$ |
| \_\_init\_\_ | Initialize cvqIon instance | Mass $m$<br>Charge $q$<br>Char. length $d$<br>Init. wavefunction $\psi_0$<br>Num. Fock states $N$<br>Mode frequencies<br>Heating rate<br>Dephasing rate |
| calc\_expectations | Calculate expectation values over history | List of operators |
| expectation\_plots | Plot expectation values over history | List of operators<br>Operator names<br>Time scale |
| create\_wigner\_plot | Generate Wigner plots of wavefunction | $x$-dimension range<br>$p$-dimension range |
| cont\_wigner\_plot | Generate an animation showing the evolution of the Wigner plot over time | List of plot states<br>$x$-dimension range<br>$p$-dimension range<br>Figure and axes to plot on<br>Title for figure |
| plot\_fock | Plot distributions over Fock states | N/A |
| fidelity | Compute fidelity of noisy w.r.t. noiseless | N/A |

We primarily made use of the carrier_evo and cntrl_disp_evo functions to construct the desired QSPI sensing states, as well as the cat states, and then the create_wigner_plot function in our plotting code in order to visualize these states.

### qspi\_phase\_learning.py

The cvqi\_phase\_learning.py code file provides code for learning the optimal QSPI phases given the appropriate objective function.  Its functions are outlined in the table below.

| Function Name | Description | Input Parameters |
|-|-|-|
| calc\_qsp\_coeff | Calculates $f$ and $g$ coefficients given phases | Phases |
| calc\_qsp\_coeff\_tensor | Calculates tensors of $f$ and $g$ coefficients given phases | Phases |
| prob\_exact | Calculates the probability of error $p_{\rm err}$ given phases | Phases<br>Threshold $\beta_{\rm th}$<br>Scale parameter $K$ |
| loss\_fn\_exact | Calculates the exact value of the loss function given phases | Phases<br>Threshold $\beta_{\rm th}$<br>Scale parameter $K$<br>Variance of state $\sigma$<br>Boolean flag indicating whether called from callback |

Using these functions, and, in particular, the function loss\_fn\_exact, we implement in this code file machine optimization on the QSPI phases with the Nelder-Mead optimization algorithm in order to find phases that make optimal decisions about displacement of a quantum state.

The procedure to learn or fine-tune phases for any QSPI sequence length is to navigate to the end of the qspi\_phase\_learning.py file and add the desired degrees to the variable degree\_list, also setting the desired parameters for convergence in the options variable.  This alone performs the learning from a fully random initial state, which, admittedly, has the weakness that the phases might get caught in a local minimum of the loss function rather than being minimized all of the way to the global loss, which is why it might be beneficial to set the num\_trials variable to more than 1 so that the program learns the best phases beginning from multiple distinct initial conditions.  If instead of learning completely from scratch one would like to fine-tune a sequence of phases, one should initialize the variable phases0 to a Pytorch tensor of the desired initial condition prior to the line setting the variable res equal to the result of the scipy.optimize.minimize function.  All of the other variables should also be fairly self-explanatory, with the most important being K, which represents $k$, which also determines the threshold $\beta_{\rm th}$, represented by the variable beta\_th.

### cvqi\_and\_cat\_wigner\_plots.py

The cvqi\_and\_cat\_wigner\_plots.py code file is used only to generate the CVQI sensing state and cat state Wigner plots given the learned QSPI phases.  It primarily utilizes the cvqi.py code for the cvqiIon class to do so and does not have any functions of its own.

### plot\_qspi\_sensing\_state\_response.py

The plot\_qspi\_sensing\_state\_response.py code file is used only to plot the qubit response functions for the learned QSPI phases.  As such, it only has a couple of new functions for generating the arrays used to construct these plots, while the rest are copies of the functions in qspi_phase_learning.py.  The new functions are outlined in the table below.

| Function Name | Description | Input Parameters |
|-|-|-|
| signal\_poly\_prob\_grid | Calculates the probabilities of the qubit response function in the range from $-\frac{\pi}{2K}$ to $\frac{\pi}{2K}$ given the number of grid units to include | Phases<br>Scale parameter $K$<br>Variance of state $\sigma$<br>Number of grid units to include |
| signal\_poly\_prob\_grid\_qsp\_partial | Calculates the probabilities of the qubit response function in a subset of the range from $-\frac{\pi}{2K}$ to $\frac{\pi}{2K}$ given the number of grid units to include | Phases<br>Scale parameter $K$<br>Variance of state $\sigma$<br>Number of grid units to include<br>Proportion of the range to include<br>Boolean indicating whether to begin at $-\frac{\pi}{2K}$ or end at $\frac{\pi}{2K}$ |

Using these two functions, we calculate all of the values necessary to generate the plots for our paper using the optimal QSPI phases learned from the code given in cvqi\_phase\_learning.py.

## Citing this repository

To cite this repository please include a reference to [our paper](https://www.google.com)

## History

- v0.0.0 The initial commit with just the code referenced in the paper
