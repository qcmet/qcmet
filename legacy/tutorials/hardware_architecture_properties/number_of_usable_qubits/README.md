# Number of usable qubits

In this directory we have the code for verifying the number of usable qubits in a quantum computer.

### Parameters

You will need to run the `number_of_usable_qubits.ipynb` notebook.

There are parameter that can be adjusted, such as:

- `reported_n_qubits` - the number of qubits provided by the hardware manufacturer.

- `device_name` - the name of the device to use. Note that simulators do not have a maximum number of usable qubits, hence the testing loop does not terminate.

### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will print out the number of usable qubits in the device.


