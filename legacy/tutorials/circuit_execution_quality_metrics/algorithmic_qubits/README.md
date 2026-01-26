# Algorithmic qubits

In this directory we have the code for the algorithmics qubits metric, #AQ. This metric corresponds to the largest number of qubits that can run a set of selected quantum algorithms successfully according to a specific success criterion.

### Setup and usage

This metric follows the definition of #AQ version 1.0 created by IonQ, and defined in their [repository](https://github.com/ionq/QC-App-Oriented-Benchmarks). The code for this metric is provided in the directory `./code/` and is a clone of the repository provided by IonQ. We provide the software as part of this repository for ease of access for the user. Please note that the software provided by here is frozen in time and may not necessarily be kept up to date with the original IonQ repository. 

In order to run the software please first go to the `./code/_setup/` directory and click on the relevant folder for qiskit, cirq or braket. Once you have completed the setup, you may copy the relevant `./code/benchmarks-<setup-version>.ipynb.template` file, where `<setup-version>` is replaced with qiskit, braket or cirq, and ensure that the file ends in an ipynb format.

For any further queries about the software, you may go through the readme files provided in the `./code/` folder written by IonQ.

To view example output, please look at the saved results in `./code/aq-qiskit.ipynb`. The value for the #AQ metric is given at the bottom of the python notebook.

### Notes on required environment

The #AQ software requirements may clash with the requirements for this current repository and as such it is recommended that the user creates a separate python environment to run this metric. The environment should be set up as specified by the files in `./code/_setup/`.

