# Pairwise connectivity

In this directory we have the code for verifying the pairwise qubit connectivity in a quantum computer.

### Core dependencies

 - `qiskit==0.45.2` - for constructing the circuit

 - `braket` - for submitting circuits

 - `qiskit-braket-provider` - for interfacing qiskit and braket


### Parameters

You will need to run the `pairwise_connectivity.ipynb` notebook.

There are parameter that can be adjusted, such as:

- `reported_connectivities` - the list of qubit connectivities as reported by the hardware manufacturer. 

- `device_name` - the name of the device to use.

### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will print out the validity of the reported qubit connectivities.


