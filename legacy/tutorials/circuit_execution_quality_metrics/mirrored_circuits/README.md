# Mirrored circuits

In this directory we will have the code for mirrored circuits benchmark.


### Parameters

To run the mirrored circuit benchmark, you will need to run the `hubbard_model_simulation.ipynb` notebook.

There are parameters that can be adjusted, such as:

- `w` - the number of qubits, denoted as w for 'width' of circuit

- `m` - the number of Clifford circuit layers that form the base circuit $C$

- `base_circuit` - (optional) a custom base circuit $C$ instead of generating it randomly

- `k` - the number of mirrored circuits to generate and run the benchmark for

- `shots` - the number of measurement shots

- `device_name` - the name of the (AWS) device to use. Default to "noisy_sim" for noisy simulations

- `noise_model` - an optional `qiskit_aer.noise.NoiseModel` to use for noisy simulations


### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will run the mirrored circuits benchmark with a set of pre-defined parameters and an randomly generated example base circuit.

