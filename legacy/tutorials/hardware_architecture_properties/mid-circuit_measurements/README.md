# Capability to perform mid-circuit measurements

In this directory we have the code for testing if a device supports mid-circuit measurements.

### Parameters

You will need to run the `mid-circuit_measurements.ipynb` notebook. There are no parameters for this notebook.

### Usage

As the notebook is set up now, if the required dependencies are installed, you may run the notebook with jupyter notebook by clicking on 'Run All'.

This will create an example circuit containing mid-circuit measurement, and run it using a custom noise model. In qiskit, mid-circuit measurements are supported, hence you will see that the result show equal probabilities for the following two sets of states: {$\ket{0},\ket{00}$} and {$\ket{1}, \ket{10}$}, while other readouts appear sparingly due to the noise.

There is no `measurement()` command for AWS Braket circuits, and all measurements are performed automatically at the end of circuits, hence mid-circuit measurements are not possible.  


