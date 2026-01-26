# Cycle benchmarking

In this directory we have the code for running cycle benchmarking.

### Core dependencies

In addition to the base requirements for this repository, this metric requires pytket, which is used for randomized compiling. You can install it using the requirements file  `cb_requirements.txt` or install the following package listed below using pip.
 - `pytket` - for randomised compilation


### Parameters

To run the benchmark, you will need to run the `cycle_benchmarking_composite_process_fidelity.py` file.

There are parameters that can be adjusted, such as:

- `n_qubits` - the number of qubits to run RB on. Choose from 1 or 2

- `num_shots` - the number of shots

- `num_rand_sequences_L` - the number of randomised circuits to run for each pauli subspace

- `total_num_qubits` - the total number of qubits in the tested device. For noisy simulations, this number can be set to be equal to `n_qubits`

- `method` - the method to calculte the cycle benchmarking composite process fidelity. Choose between "fit" and "ratio"

- `device_name` - the name of the (AWS) device to use. Default to "noisy_sim" for noisy simulations

- `noise_model` - an optional `qiskit_aer.noise.NoiseModel` to use for noisy simulations

### Usage

As the script is set up now, if the required dependencies are installed, you may run the python script from the command line using the command `python3 cycle_benchmarking.py`. The output will give something like `Composite process fidelity: 0.9800`.
