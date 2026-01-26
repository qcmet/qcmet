# Variational Quantum Eigensolver (VQE)

In this directory we have the benchmarking code for variational quantum eigensolver (VQE).

### Core dependencies
In addition to the `base_requirements.txt` in the root of this directory you must also install the requirements file found in this directory called `vqe_requirements.txt`. We also list the requirements in this file below.

 - `qiskit-nature==0.7.2` - for generating the initial state as a qiskit circuit 

 - `openfermion==1.6.1` - for mapping the hamiltonian to qubits

The software has only been tested to run correctly with the versions specified above.

### Parameters

To run the VQE metric, you will need to run the `vqe.ipynb` notebook. Details of the methdology is explained in the notebook as well as the metrics document.

There are parameters that can be adjusted, such as:

- `nsites` - the number of Fermionic sites.

- `t` - the hopping integral.

- `U` - the onsite energy.

- `num_layers_vha` - the number of layers to apply of the Hamiltonian variational ansatz.

- `num_trials` - The number of VQE circuits to run to get statistics on execution quality.

- `device_name` - the name of the (AWS) device to use. Default to "noisy_sim" for noisy simulations.

- `noise_model` - an optional `qiskit_aer.noise.NoiseModel` to use for noisy simulations.

 The parameters are pre-set to `nsites`= 3, `t`=1, `U`=2, `num_layers_vha`= 1 and `num_trials`=100.

### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will run the Hubbard model simulation with qiskit using both a noiseless state vector simulator and a noisy simulator.

The average energy difference per-site between the noiseless state vector simulator and the noisy sumulations energy calculations will be computed. This together with the details of the simulations will be saved into the data subdirectory. Each run is in its own unique folder which is created depending on the time it was created.



