# Quantum Fourier Transform (QFT)

In this directory we have the benchmarking code for quantum fourier transform (VQE).

### Parameters

To run the QFT metric, you will need to run the `qft.ipynb` notebook.

There are parameters that can be adjusted, such as:

- `n_qubits` - the number of qubits to run on.

- `device_name` - the name of the (AWS) device to use. Default to "noisy_sim" for noisy simulations.

- `noise_model` - an optional `qiskit_aer.noise.NoiseModel` to use for noisy simulations.

### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will run the QFT benchmark on the specified device. A normalised fidelity between the two obtained probability distributions will be calculated. This together with the details of the simulations will be saved into the data subdirectory. Each run is in its own unique folder which is created depending on the time it was created.

In the subdirectory, there will be a plot output for the experiemnt.

