# Quantum Signal Processing Interferometry

## Introduction

In this work, we make use of a generalization of standard qubit [Quantum signal processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501) (QSP) to the case of an ancilla qubit coupled to a bosonic oscillator to perform signal processing of the bosonic quadrature operators (**Bosonic QSP**) based on a natural physical block-encoding of $e^{-i h(\hat{x}, \hat{p})}$ inside a signal unitary $U = e^{-i h(\hat{x}, \hat{p}) \sigma_z}$, where $h(\hat{x}, \hat{p})$ is an analytic function (often truncated to a finite-order polynomial function) of the bosonic position ($\hat{x}$) and momentum ($\hat{p}$) operators.  We build on top of such bosonic QSP the construction of a bosonic **QSP Interferometry** protocol and utilize it to perform quantum sensing of displacements on the bosonic mode.  In particular, for our signal unitary $U$, we choose $h(\hat{x}, \hat{p}) = -\kappa \hat{x}$ for a constant momentum kick $\kappa$.  This means our bosonic QSP signal operator is essentially a controlled displacement operation on the bosonic mode with the ancilla sensing qubit taken as the control, denoted by $c\mathcal{D}(\cdot)$, alternated with a qubit rotation on the ancilla qubit as the signal processing unitary.  We show that such a construction can realize an arbitrary degree- $d$ real Laurent polynomial transformation on $\omega(x) = e^{i\kappa\hat{x}}$ with a depth $\mathcal{O}(d)$ circuit.  This allows us to prepare a wide range of continuous-variable oscillator states with significant flexibility.

Such bosonic QSP enables more efficient and specifically-tailored quantum sensing protocols beyond simple parameter estimation.  In particular, using the binary measurement outcome on the ancilla qubit, we demonstrate Heisenberg-like scaling in performing quantum decisions for a unitary displacement channel.

## Using the code in this repository

This repository includes four code files, namely <strong><code>cvqi.py</code></strong>, <strong><code>qspi\_phase\_learning.py</code></strong>, <strong><code>cvqi\_and\_cat\_wigner\_plots.py</code></strong>, and <strong><code>plot\_qspi\_sensing\_state\_response.py</code></strong>.  Each code file serves a unique but important role for this work and its visualizations, so we will briefly describe each in a separate section of this README.

Note that in order to utilize this code, there are a few basic requirements regarding necessary packages.  These are outlined in the document <strong><code>src/requirements.txt</code></strong> of this repository.  Note that to install these required packages, it is necessary only to navigate to the directory in which the text file <strong><code>requirements.txt</code></strong> is located and then enter the line `pip install -r requirements.txt` in the terminal.  This will install all of the required versions of the packages for running the code for QSPI phase learning and visualization.

### cvqi.py

The <strong><code>cvqi.py</code></strong> code file provides code for simulating continuous-variable quantum computing, or CVQI, operations, on a qumode, as well as code for hybrid operations with an ancilla qubit.  The functions are outlined in the table below.

| Function Name | Description | Input Parameters |
|-|-|-|
| \_\_apply\_h | Evolution under any Hamiltonian | Hamiltonian $H$<br>Operation time $t$<br>Master Equation solver name<br>Additional arguments dict. |
| carrier\_evo | n'th blue sideband transition evolution | Mode number $j$<br>Rotation axis $\phi$<br>Rotation angle $\theta$<br>Operation time $t$ |
| carrier\_evo\_exact | n'th blue sideband transition evolution simulated exactly by evaluating $e^{-iHt}$ | Mode number $j$<br>Rotation axis $\phi$<br>Rotation angle $\theta$<br>Operation time $t$ |
| free\_evo | Free Hamiltonian evolution | Operation time $t$ |
| disp\_evo | Displacement Hamiltonian evolution | Mode number $j$<br>Displacement parameter $\alpha$<br>Operation time $t$ |
| disp\_evo\_exact | Displacement Hamiltonian evolution simulated exactly by evaluating $e^{-iHt}$ | Mode number $j$<br>Displacement parameter $\alpha$<br>Operation time $t$ |
| sq\_evo | Squeezing Hamiltonian evolution | Mode number $j$<br>Parametric amp. strength $g$<br>Modulation drive phase $\theta$<br>Operation time $t$ |
| bs\_evo | Beamsplitter Hamiltonian evolution | First mode number $j$<br>Second mode number $k$<br>Beamsplitter strength $g$<br>Operation time $t$ |
| el\_drive | Time-dependent classical electric drive field evolution | Amplitude of E-field $A$<br> Phase of E-field $\phi$<br>E-field drive frequency $f$<br>Mode number $j$<br>Amplitude modulation of E-field $e$<br>Operation time $t$<br>QuTiP Solver *solver*<br>Boolean for use of Blackman window vs. linear ramp-up and ramp-down<br>Ramp-up proportion *ramp\_up*<br>Ramp-down proportion *ramp\_down* |
| phase\_evo | Phase-shift Hamiltonian evolution | Mode number $j$<br>Phase $\phi$<br>Operation time $t$ |
| rsb\_evo | Red side-band Hamiltonian evolution | Mode number $j$<br>Side-band order $n$<br>Phase $\phi$<br>Operation time $t$ |
| bsb\_evo | Blue side-band Hamiltonian evolution | Mode number $j$<br>Side-band order $n$<br>Phase $\phi$<br>Operation time $t$ |
| stark\_evo | AC Stark Hamiltonian evolution | Mode number $j$<br>Operation time $t$ |
| kerr\_evo | Cross-Kerr Hamiltonian evolution | Mode number $j$<br>Kerr nonlinearity $\chi$<br>Operation time $t$ |
| cntrl\_disp\_evo | Controlled displacement Hamiltonian evolution | Mode number $j$<br>Displacement parameter $\alpha$<br>Spin pauli phase $\phi$<br>Operation time $t$ |
| cntrl\_disp\_evo\_exact | Controlled displacement Hamiltonian evolution simulated exactly by evaluating $e^{-iHt}$ | Mode number $j$<br>Displacement parameter $\alpha$<br>Spin pauli phase $\phi$<br>Operation time $t$ |
| \_\_init\_\_ | Initialize cvqIon instance | Mass $m$<br>Charge $q$<br>Char. length $d$<br>Init. wavefunction $\psi_0$<br>Num. Fock states $N$<br>Mode frequencies<br>Heating rate<br>Dephasing rate |
| calc\_expectations | Calculate expectation values over history | List of operators |
| expectation\_plots | Plot expectation values over history | List of operators<br>Operator names<br>Time scale |
| create\_wigner\_plot | Generate Wigner plots of wavefunction | $x$-dimension range<br>$p$-dimension range |
| cont\_wigner\_plot | Generate an animation showing the evolution of the Wigner plot over time | List of plot states<br>$x$-dimension range<br>$p$-dimension range<br>Figure and axes to plot on<br>Title for figure |
| plot\_fock | Plot distributions over Fock states | N/A |
| fidelity | Compute fidelity of noisy w.r.t. noiseless | N/A |

We primarily make use of the carrier\_evo\_exact and cntrl\_disp\_evo\_exact functions to construct the desired QSPI sensing states, as well as the cat states, and we also use the disp\_evo\_exact function for implementing the displacement operation to be sensed in the evaluation of the full protocol.  We then utilize the create\_wigner\_plot function in our plotting code in order to visualize these states.

### qspi\_phase\_learning.py

The <strong><code>cvqi\_phase\_learning.py</code></strong> code file provides code for learning the optimal QSPI phases given the appropriate objective function.  Its functions are outlined in the table below.

| Function Name | Description | Input Parameters |
|-|-|-|
| calc\_qsp\_coeff | Calculates $f$ and $g$ coefficients given phases | Phases |
| calc\_qsp\_coeff\_tensor | Calculates tensors of $f$ and $g$ coefficients given phases | Phases |
| prob\_exact | Calculates the probability of error $p_{\rm err}$ given phases | Phases<br>Threshold $\beta_{\rm th}$<br>Scale parameter $\kappa$ |
| loss\_fn\_exact | Calculates the exact value of the loss function given phases | Phases<br>Threshold $\beta_{\rm th}$<br>Scale parameter $\kappa$<br>Variance of state $\sigma$<br>Boolean flag indicating whether called from callback |

Using these functions, and, in particular, the function loss\_fn\_exact, we implement in this code file machine optimization on the QSPI phases with the Nelder-Mead optimization algorithm in order to find phases that make optimal decisions about displacement of a quantum state.

The procedure to learn or fine-tune phases for any QSPI sequence length is to navigate to the end of the <strong><code>qspi\_phase\_learning.py</code></strong> file and add the desired degrees to the variable degree\_list, also setting the desired parameters for convergence in the options variable.  This alone performs the learning from a fully random initial state, which has the weakness that during optimization, the phases might get stuck in a local minimum of the loss function rather than being minimized all of the way to the global loss.  As such, it might be beneficial to set the num\_trials variable to more than 1 so that the program learns the best phases beginning from multiple distinct initial conditions.  If instead of learning completely from scratch one would like to fine-tune a sequence of phases, one should initialize the variable phases0 to a Pytorch tensor of the desired initial condition prior to the line setting the variable res equal to the result of the scipy.optimize.minimize function.  All of the other variables should also be fairly self-explanatory, with the most important being $\kappa$, the scale parameter represented by the variable K, which also determines the threshold $\beta_{\rm th}$, represented by the variable beta\_th.

### cvqi\_and\_cat\_wigner\_plots.py

The <strong><code>cvqi\_and\_cat\_wigner\_plots.py</code></strong> code file is used only to generate the CVQI sensing state and cat state Wigner plots given the learned QSPI phases.  It primarily utilizes the <strong><code>cvqi.py</code></strong> code for the cvqiIon class to do so and does not have any functions of its own.

### plot\_qspi\_sensing\_state\_response.py

The <strong><code>plot\_qspi\_sensing\_state\_response.py</code></strong> code file is used only to plot the qubit response functions for the learned QSPI phases.  As such, it only has a couple of new functions for generating the arrays used to construct these plots, while the rest are copies of the functions in <strong><code>qspi_phase_learning.py</code></strong>.  The new functions are outlined in the table below.

| Function Name | Description | Input Parameters |
|-|-|-|
| signal\_poly\_prob\_grid | Calculates the probabilities of the qubit response function in the range from $-\frac{\pi}{2\kappa}$ to $\frac{\pi}{2\kappa}$ given the number of grid units to include | Phases<br>Scale parameter $\kappa$<br>Variance of state $\sigma$<br>Number of grid units to include |
| signal\_poly\_prob\_grid\_qsp\_partial | Calculates the probabilities of the qubit response function in a subset of the range from $-\frac{\pi}{2\kappa}$ to $\frac{\pi}{2\kappa}$ given the number of grid units to include | Phases<br>Scale parameter $\kappa$<br>Variance of state $\sigma$<br>Number of grid units to include<br>Proportion of the range to include<br>Boolean indicating whether to begin at $-\frac{\pi}{2\kappa}$ or end at $\frac{\pi}{2\kappa}$ |

Using these two functions, we calculate all of the values necessary to generate the plots for our paper using the optimal QSPI phases learned from the code given in <strong><code>cvqi\_phase\_learning.py</code></strong>.

## Citing this repository

To cite this repository please include a reference to [our paper](https://arxiv.org/pdf/2311.13703.pdf)

## History

- v0.0.0 The initial commit with just the code referenced in the paper
