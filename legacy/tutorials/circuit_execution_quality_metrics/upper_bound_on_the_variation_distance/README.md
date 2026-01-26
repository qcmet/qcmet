# Upper bound on the variation distance within an accreditation protocol


In this directory we have the code for quantum accreditation (AP), which provides an upper bound on the variation distance between the probability distribution of the experimental outputs of a noisy quantum circuit and its noiseless counterparts.

### Parameters

To run the quantum accreditation protocol, you will need to run the `accreditation.ipynb` python notebook.

There are parameters that can be adjusted, such as:
- `target_circuit` - a circuit to run the quantum accreditation protocol for. An example is given for a four-qubit circuit

- `mu` - the desired accuracy of AP 

- `eta` - the desired confidence of AP 

- `noise_model` - the noise model for simulating a noisy quantum computer

- `device_name` - the name of the (AWS) device to use. Default to "noisy_sim_with_shots" for noisy simulations

### Usage

As the script is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will run the quantum accreditation test for the example circuit. The output is an upper bound on the variation distance between the probability distribution of the experimental outputs of a noisy target quantum circuit and its noiseless counterparts.
