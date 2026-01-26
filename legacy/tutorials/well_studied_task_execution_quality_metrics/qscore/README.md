# Q-Score benchmarking

In this directory we have the code for benchmarking quantum algorithms for optimisation problems using Q-Score.


### Parameters

To run the Q-Score benchmarking, you will need to run the `qscore.ipypy` notebook.

There are parameters that can be adjusted, such as:
- `min_n` - the minimum size of the MaxCut problem to be solved, corresponding to the minimum number of qubits.

- `max_n` - the maximum size of the MaxCut problem to be solved, corresponding to the maximum number of qubits.

- `depth` - the depth of the QAOA ansatz.

- `noise_model` - the noise model for simulating a noisy quantum computer.


### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will run the Q-Score benchmark with qiskit using a noisy simulator that simulates the noise levels in the IBMQ Kolkata device. The program will run the QAOA algorithm to solve MaxCut problems up to `max_n`.

For each problem size `n`, it runs 100 different graphs of MaxCut and computes a score $\beta(n)$ from the average cost that can be achieved by the (simulated) device. If the ratio of this score to a constant random best score is greater than 0.2, then the device has succeeded in the size `n` and the program outputs 'Success'; otherwise, it outputs 'Fail'. The largest problem size where the device succeeded is the Q-Score. If all runs succeeded, then `max_n` should be increased to test for larger problem sizes.
