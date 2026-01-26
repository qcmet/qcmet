# QCMet - Quantum Computing Metrics and Benchmarks
## Software repository for ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717)

### Contents
1. [Summary of reference article](#1-summary-of-reference-article)
2. [Overview](#2-overview)
3. [Running the software](#3-running-the-software)
4. [Notes on dependencies](#4-notes-on-dependencies)

QCMet is the software accompanying the article ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717). It is licensed under the Apache License, Version 2.0. Copyright information can be found in NOTICE, and the license itself in LICENSE. If you use this code in your work, please cite the accompanying article: 


 - D. Lall, A Agarwal, W. Zhang, L. Lindoy, T. Lindström, S. Webster, S. Hall, N. Chancellor, P. Wallden, R. Garcia-Patron, E. Kashefi, V. Kendon, J. Pritchard, A. Rossi, A. Datta, T. Kapourniotis, K. Georgopoulos, I. Rungger, _A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software_, [arXiv:2502.06717](https://doi.org/10.48550/arXiv.2502.06717).

Most of the metrics and benchmarks in the aformentioned article have an associated software in this QCMet repository.
The software has been designed so that, where possible, one may evaluate each metric using a circuit submitter interface that allows for submission to an emulator, or real quantum computers that are accessed using services such as Amazon Braket.

In the `tutorials` directory, you can find subdirectories for each metric, which contains a `README.md` with instructions on how to run the calculation.

###  1. Summary of reference article
This software repository is designed to be used in conjunction with the article ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717). This article explains the metrics contained in this repository. The motivation and overview of metrics and benchmarks is provided in the introduction of that article, and a summarizing overview in part II. This part contains a motivation for each of the selected metrics and why they have been categorized into categories M1 to M10. If you wish for a short description of the metrics and benchmarks without going into too much depth, you may read section 2 of part II. This section also contains the most relevant references pertaining to each metric. For a discussion and outlook regarding future work and standardization, you may read section 3 in part II.

If one then wishes to understand each metric more deeply, then the user should find the relevant metric in part IV. This part contains, for each metric, a definition, a description, the measurement methodology listed as a step-by-step guide, assumptions and limitations, a link to the associated software, and references.  


###  2. Overview

Together with the metric entry from the collection of metrics in ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717), the software provided in this repository is meant as a tutorial for each metric. The software in this repository is intended as guide to demonstrate to the user how each metric may be evaluated. The software has been evaluated on quantum computing emulators with general noise models, and as part of its development it was also evaluated on a number of hardware platforms. When running on a hardware platform it may need adapting to the hardware-specific capabilities and commands. 


Below is a list of the collection of metrics provided in ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717). Software is provided for the metrics linked below.

 1. [Hardware architecture properties](tutorials/hardware_architecture_properties/)  
   M1.1. [Number of usable qubits](tutorials/hardware_architecture_properties/number_of_usable_qubits/)  
   M1.2. [Pairwise connectivity](tutorials/hardware_architecture_properties/pairwise_connectivity/)  
   M1.3. [Native gate set](tutorials/hardware_architecture_properties/native_gates/)  
   M1.4. [Capability to perform mid-circuit measurements](tutorials/hardware_architecture_properties/mid-circuit_measurements/)  
 2. [Qubit quality metrics](tutorials/qubit_quality_metrics/)  
   M2.1. [Qubit relaxation time ($T_1$)](tutorials/qubit_quality_metrics/t1/)  
   M2.2. [Qubit dephasing time ($T_2$)](tutorials/qubit_quality_metrics/t2/)  
   M2.3. [Idle qubit purity oscillation frequency](tutorials/qubit_quality_metrics/idle_qubit_purity_oscillation/)  
 3. [Gate execution quality metrics](tutorials/gate_execution_quality_metrics/)  
   M3.1. [Gate set tomography based process fidelity](tutorials/gate_execution_quality_metrics/gst_based_gate_execution_quality_metrics/)  
   M3.2. [Diamond norm of a quantum gate](tutorials/gate_execution_quality_metrics/gst_based_gate_execution_quality_metrics/)  
   M3.3. [Clifford randomized benchmarking](tutorials/gate_execution_quality_metrics/randomized_benchmarking/clifford_randomized_benchmarking/)  
   M3.4. [Interleaved Clifford randomized benchmarking](tutorials/gate_execution_quality_metrics/randomized_benchmarking/interleaved_clifford_randomised_benchmarking/)  
   M3.5. [Cycle-benchmarking composite process fidelity](tutorials/gate_execution_quality_metrics/cycle_benchmarking_composite_process_fidelity/)  
   M3.6. [Over- or under- rotation angle](tutorials/gate_execution_quality_metrics/over_under_rotations/)  
   M3.7. [State preparation and measurement fidelity](tutorials/gate_execution_quality_metrics/gst_based_gate_execution_quality_metrics/)  
 4. [Circuit execution quality metrics](tutorials/circuit_execution_quality_metrics/)  
   M4.1. [Quantum volume](tutorials/circuit_execution_quality_metrics/quantum_volume/)  
   M4.2. [Mirrored circuits average polarization](tutorials/circuit_execution_quality_metrics/mirrored_circuits/)  
   M4.3. [Algorithmic qubits](tutorials/circuit_execution_quality_metrics/algorithmic_qubits/)  
   M4.4. [Upper bound of the variation distance](tutorials/circuit_execution_quality_metrics/upper_bound_on_the_variation_distance/)  
 5. [Well studied task execution quality metrics](tutorials/well_studied_task_execution_quality_metrics/)  
   M5.1. [Variational quantum eigensolver (VQE) metric](tutorials/well_studied_task_execution_quality_metrics/vqe/)  
   M5.2. [Quantum approximate optimization algorithm (QAOA) metric](tutorials/well_studied_task_execution_quality_metrics/qscore/)  
   M5.3. [Fermi-Hubbard model simulation (FHMS) metric](tutorials/well_studied_task_execution_quality_metrics/hubbard_model_simulation/)  
   M5.4. [Quantum Fourier transform (QFT) metric](tutorials/well_studied_task_execution_quality_metrics/qft/)  
 6. [Speed metrics](tutorials/speed_metrics/)  
   M6.1. [Time take to execute a general single- or multi-qubit gate](tutorials/speed_metrics/time_taken_to_execute_a_general_single_or_multi_qubit_gate/)  
   M6.2. [Time to measure qubits](tutorials/speed_metrics/time_to_measure_qubits/)  
   M6.3. [Time to reset qubits](tutorials/speed_metrics/time_to_reset_qubits/)  
   M6.4. Overall device speed on reference tasks   [<span style="color:orange">*No software*</span>]  
 7. [Stability metrics](tutorials/stability_metrics/)  
   M7.1. [Standard deviation of a specific metric evaluated over a time interval](tutorials/stability_metrics/amount_of_fluctuations_of_metrics_over_time/)  
 8. [Metrics for quantum annealers](tutorials/quantum_annealers/)  
   M8.1. [Single qubit control errors](tutorials/quantum_annealers/single_qubit_control_errors/)  
   M8.2. Size of the largest mappable fully connected problem  [<span style="color:orange">*No software*</span>]  
   M8.3. Dimensionless sample temperature  [<span style="color:orange">*No software*</span>]  
 9. Metrics for boson sampling devices  [<span style="color:orange">*No software*</span>]  
   M9.1. Hardware characterization and model as metrics  [<span style="color:orange">*No software*</span>.]    
   M9.1. Quantum advantage demonstration as metric  [<span style="color:orange">*No software*</span>]  
 10. Metrics for neutral atom devices    [<span style="color:orange">*No software*</span>]  
   M10.1. Trap lifetime  [<span style="color:orange">*No software*</span>]  
   M10.2. Reconfigurable connectivity [<span style="color:orange">*No software*</span>]  


###  3. Running the software
Most metrics contain a parameter for choosing which simulator or quantum hardware device you would like to run on. The default is to run on a local simulator with a noise model described in the article.

#### The circuit submitter helper

A circuit submitter helper has been integrated into many of the tutorials. Using this, one may select a variety of backends. 

The circuit submitter is used in all of the metrics to submit to a backend. The circuit submitter is called as follows:
```python
from _helpers.circuit_submitter import CircuitSubmitter

device_name = "noisy_sim"
submitter = CircuitSubmitter("<benchmark_name>", device_name)
```

The submitter must be initialized before being able to submit any circuits to the desired backend. The backends can be chosen by specifying the device name.

The emulator devices available and their names are defined as follows:
| Emulator                           | device_name            | Notes          |
|------------------------------------|------------------------|----------------|
| Braket local simulator             | "simulator"            | Default AWS Braket LocalSimulator |
| Noiseless density matrix emulator  | "noiseless_sim"        | Qiskit based density matrix simulator with no shot noise|
| Noisy emulator with no shot noise  | "noisy_sim"            | Qiskit based emulator with noise specified in `/_helpers/noise_model.py` set to a specific seed, but with no shot noise, so result is deterministic|
| Noisy emulator with shots          | "noisy_sim_with_shots" | Qiskit based emulator with noise specified in `/_helpers/noise_model.py`|


If a custom noise model is required, you may specify one after initialising the `CircuitSubmitter`. Please note, that the custom noise model expected is a `qiskit_aer.noise.NoiseModel` object.

```Python
submitter.backend.device.set_noise_model( noise_model = user_specified_qiskit_noise_model)
```
Once the submitter has been initialized, then the benchmark circuits can be executed on the chosen backend with the following commands:

```Python
submitter.submit_circuits(shots, qasm_strs=list_of_qasm_circs)
```

When submitting circuits, please note that only one of following arguments to submit circuits should be used:
  - braket_circuits: a list of `braket.Circuit` objects.
  - qasm_strs: a list of qasm strings, each representing one circuit.
  - qasm_paths: a list of qasm file paths, each representing one circuit.

Additional arguments exist that the user can specify:
  - shots: number of shots for each circuit.
  - verbatim: whether to force verbatim circuits with no optimisation.
  - skip_asking: If set to false, the submitter waits for user to responds Y/n if the circuits should be submitted. This is useful in the context of hardware runs where it is useful to first have cost estimation before running the circuit list.
  - skip_transpilation: whether to skip transpilation of circuits, to directly run the circuits. This is only possible with qasm circuits is the backend device is "noisy_sim" or "noisy_sim_with_shots". 


#### Hardware runs

At the time of writing the article, the circuit submitter was used to run benchmarks on hardware hosted on AWS Braket. For more details on Braket and creating your own backend for the circuit submitter, please see [this readme](_helpers/README_circuit_submitter_and_hardware_runs.md).

In general, there are a few considerations that must be made when running on hardware.

1. Different quantum computers may have different native gates. This means that whilst the benchmarking circuits are designed to generate circuits using a specific gate set, they may not be able to be executed on some hardware without compilation to that specific set of native gates.
2. Often, changing parameters in a metric can change the number of circuits required to run as part of the metric. This will affect the run time and also change the costs associated with running the benchmark. In our CircuitSubmitter interface, we allow for the estimation of costs of submitting circuits by using pricing set by the backend.
3. Whilst running on an emulator means circuits can be executed instantly, often when running on hardware, you must input your circuits in a queue and wait for execution. This asynchronous execution means that the circuit submitter must query the hardware at a set frequency to check if the circuits have been submitted, and then retrieve the results.
4. Often in emulator runs, all-to-all connectivity of qubits is assumed. However, when running on hardware, this is not always the case. In addition to compiling to the native gates of the hardware, the circuits must be compiled such that they also respect the connectivity of multi-qubit gates available on the device.


###  4. Notes on dependencies

The following applies for all metrics except algorithmic qubits. Please read the algorithmic qubits section below for details on its dependencies. 

All code is tested under python version 3.10.13. Compatibility with other python versions has not been tested and cannot be guaranteed.

All top-level package dependencies except `mpi4py` are listed in `full_requirements.txt`. 

However, we also have a base level of requirements that most metrics run off, and some package dependencies are only relevant to specific metrics and are not used elsewhere. This base requirements list is given in the file `base_requirements.txt`.

The other more sparingly used packages are listed in the following requirements files for specific corresponding metrics:
- `cb_requirements.txt` - for cycle benchmarking
- `mpi_requirements.txt` - for running GST with MPI
- `pygsti_requirements.txt` - for running GST or randomized benchmarking with PyGSTi
- `qscore_requirements.txt` - for running Quantum approximate optimization algorithm (QAOA) metric
- `vqe_requirements.txt` - for running the VQE metric.

All of the above specific requirements, EXCULDING `mpi_requirements.txt`, are included in the `full_requirements.txt` file.

You may simply install all required packages using the following command, which will automatically install also all the sub-dependencies of each package.
```
pip install -r full_requirements.txt
```

Note that, you will need a working MPI implementation to successfully install `mpi4py` with pip. If you do not have MPI already configured, you may try to set up the python environment using conda instead, and install `mpi4py` via conda. 

As an alternative to first installing python version 3.10.13 and then installing the dependencies with pip, there is also a `conda_environment.yml` file provided, which can be used to set up the python environment in one go with conda using the following command:
```
conda env create -f conda_environment.yml
```  
This will install all the dependencies listed in `full_requirements.txt`. However, `mpi4py` must be installed separately. 

The above dependencies have been tested to work on Linux devices. In order to run on Windows, you may need to enable long paths for python. When running on non-x86 computers such as Apple silicon Mac devices, it is recommended to avoid conda and instead use the pip requirements.

#### Algorithmic qubits
The above dependencies are valid for all tutorials except algorithmic qubits. 

Since the algorithmic qubits metric is defined by its implementation in [the IonQ repository](https://github.com/ionq/QC-App-Oriented-Benchmarks) we have cloned the repository to the folder `tutorials/circuit_execution_quality_metrics/algorithmic_qubits` for completeness, as is permitted by the Apache 2.0 license of the algorithmic qubits software. 

As a result, the algorithmic qubits software requirements may clash with the requirements for this current repository and as such it is recommended that the user creates a separate python environment to run this metric. The environment should be set up as specified by the files in `tutorials/circuit_execution_quality_metrics/algorithmic_qubits/code/_setup`.

