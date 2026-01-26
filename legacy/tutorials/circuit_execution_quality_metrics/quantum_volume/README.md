# Quantum Volume

In this directory we have the code for quantum volume (QV).


### Parameters

To run the quantum volume protocol, you will need to run the `quantum_volume.py` file.

There are parameters that can be adjusted, such as:

- `num_trials` - this specifies the number of different QV circuits for each volume. Increasing this will increase the runtime of the algorithm. 

- `num_qubits_list` - this specifies a list of the number of qubits to use, i.e. the circuit widths at which to the quantum volume test.

- `device_name` - the name of the device to use. Default to "noisy_sim" for noisy simulations. 

- `noise_model` - an optional `qiskit_aer.noise.NoiseModel` to use for noisy simulations


### Usage

As the script is set up now, if the required dependencies are installed, you may run the script simply by calling

`python3 quantum_volume.py`

This will run the quantum volume test for the chosen device and save the result to the benchmark path specified by the `CircuitSubmitter`.

In the data directory, there will be a histogram plot for the test at each width, with the histogram showing the spread of heavy output probabilites for the circuits run for the ideal and noisy simulation.
| ![Plot](./notebook_images/Ideal%20simulation%20m=5%20for%20fake_montreal_v2.png)  |![Plot](./notebook_images/Noisy%20simulation%20m=5%20for%20fake_montreal_v2.png)   |
|---|---|



There will also be a scatter plot showing the mean heavy output probability for each width.
|![ScatterPlot](./notebook_images/fake_montreal_v2%20average_heavy_output_noisy_vs_ideal.png)  |![Plot](./notebook_images/fake_mumbai_v2%20average_heavy_output_noisy_vs_ideal.png)   |
|---|---|


Note that there is a commented out block at the end of the file that allows for the use of qiskit fake backends and multiprocessing: This will run the quantum volume test for three different backends, `FakeMontreal()`, `FakeKolkata()`, and `FakeMumbai()`. The script will run the test for the set `num_trials` for each backend, and up to the `max_vol` specified.

Then the data is output in the data subdirectory. Each run is in its own unique folder which is created depending on the time it was created.

Then there will be a separate subdirectory for each different backend. 


