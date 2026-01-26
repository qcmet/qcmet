#  Idle qubit purity oscillation frequency

In this directory we have the code for benchmarking idle qubit purity oscillation frequency, which quantifies the effect of single-qubit non-Markovian noise induced coherence revivals.


### Parameters

To run the benchmark, you will need to run the `idle_qubit_purity_oscillation.ipynb` notebook.

There are parameters that can be adjusted, such as:
- `dt` - the idle time at each interval.

- `t_max` - the maximum idle time.

- `zz_crosstalk` - the strength of simulated ZZ crosstalk between two qubits.

- `shots` - the number of measurement shots.

- `device_name` - the name of the (AWS) device to use. Default to "simulator" for noiseless simulations.

- `noise_model` - an optional `qiskit_aer.noise.NoiseModel` to use for noisy simulations.


### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will run the estimation of the frequency of purity oscillations using a noisy simulator that simulates the noise levels in the IBMQ Kolkata device. The maximum of the fitted frequencies quantifies the effect of single-qubit non-Markovian noise induced coherence revivals. This value will be printed at the end of the notebook. 

