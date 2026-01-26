# QCMet - Quantum Computing Metrics and Benchmarks
> Software repository for ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717)

[![arXiv](https://img.shields.io/badge/arXiv-2502.06717-b31b1b.svg)](https://doi.org/10.48550/arXiv.2502.06717)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://qcmet.github.io/qcmet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![CI](https://github.com/qcmet/qcmet/actions/workflows/ci.yml/badge.svg)](https://github.com/qcmet/qcmet/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/deeplallnpl/3740a52e986452a0835806c50cb3d348/raw/qcmet-coverage.json)](https://github.com/qcmet/qcmet/actions)
[![Contributor's Covenant](https://img.shields.io/badge/Contributor_Covenant-v2.0_adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

QCMet (Quantum Computing Metrics and Benchmarks) is a comprehensive software framework to benchmark quantum computers.
It is accompanied by the article ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717). Most of the metrics and benchmarks in the aformentioned article have an associated software in this QCMet repository.

Read the documentation here: [https://qcmet.github.io/qcmet](https://qcmet.github.io/qcmet)

We welcome your feedback - if you have comments, suggestions, or have found an issue, please feel free to email [Deep Lall](mailto:deep.lall@npl.co.uk) or [Yannic Rath](mailto:yannic.rath@npl.co.uk).

## Contents
- [QCMet - Quantum Computing Metrics and Benchmarks](#qcmet---quantum-computing-metrics-and-benchmarks)
  - [Contents](#contents)
  - [About](#about)
  - [Key features](#key-features)
  - [Summary of reference article](#summary-of-reference-article)
  - [Installation](#installation)
    - [System requirements](#system-requirements)
    - [Basic installation](#basic-installation)
    - [Installation with optional dependencies](#installation-with-optional-dependencies)
  - [Quick start](#quick-start)
  - [Available benchmarks](#available-benchmarks)
      - [Qubit Quality Metrics](#qubit-quality-metrics)
      - [Gate Execution Quality Metrics](#gate-execution-quality-metrics)
      - [Circuit Execution Quality Metrics](#circuit-execution-quality-metrics)
      - [Well-Studied Task Execution Quality Metrics](#well-studied-task-execution-quality-metrics)
  - [Hardware runs](#hardware-runs)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)

## About

QCMet (Quantum Computing Metrics and Benchmarks) is a comprehensive software framework that provides a collection of quantum computing metrics and benchmarks. As quantum computing literature has grown extensively over the years, navigating the various metrics proposed for benchmarking quantum computer performance—from individual hardware components to entire applications can be overwhelming. The aim of QCMet is therefore to deliver a practically useful benchmarking toolkit allowing QC providers and users to benchmark devices according to well-defined metrics with the lowest possible overhead. By running our software you should be able to compare different devices.

## Key features

- **Comprehensive Metric Collection**: Implements a wide range of quantum computing metrics from literature review
- **Consistent Benchmark API**: All benchmarks inherit from `BaseBenchmark`, providing a uniform interface
- **Transparent Methodology**: Clear documentation of assumptions, limitations, and measurement procedures for each metric
- **Reproducible Implementations**: Open-source software linked to each metric ensures measurement reproducibility
- **Multiple Device Support**: Run benchmarks on ideal simulators, noisy simulators, or real quantum hardware
- **Extensible Architecture**: Easy to add new benchmarks and device backends
- **Data Management**: Built-in file management for saving circuits, measurements, and results
- **Visualization**: Automatic plotting capabilities for benchmark results
- **Qiskit Integration**: Built on Qiskit for quantum circuit generation and execution


## Summary of reference article
This software repository is designed to be used in conjunction with the article ["A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software"](https://doi.org/10.48550/arXiv.2502.06717). This article explains the metrics contained in this repository. The motivation and overview of metrics and benchmarks is provided in the introduction of that article, and a summarizing overview in part II. This part contains a motivation for each of the selected metrics and why they have been categorized into categories M1 to M10. If you wish for a short description of the metrics and benchmarks without going into too much depth, you may read section 2 of part II. This section also contains the most relevant references pertaining to each metric. For a discussion and outlook regarding future work and standardization, you may read section 3 in part II.

If one then wishes to understand each metric more deeply, then the user should find the relevant metric in part IV. This part contains, for each metric, a definition, a description, the measurement methodology listed as a step-by-step guide, assumptions and limitations, a link to the associated software, and references.  


## Installation

### System requirements

- **Python**: 3.10 or higher
- **Package Manager**: pip or uv

### Basic installation

```bash
git clone https://github.com/qcmet/qcmet.git
cd qcmet/
pip install -e .
```

### Installation with optional dependencies


For development (includes testing tools):
```bash
pip install -e .[dev]
```

For documentation building:
```bash
pip install -e .[docs]
```

## Quick start

For our quick start guide, please see our [docs](https://qcmet.github.io/qcmet/).

## Available benchmarks

QCMet currently provides the following benchmark implementations:

#### Qubit Quality Metrics
- **T1**: Qubit relaxation time measurement
- **T2**: Qubit dephasing time measurement
- **IdleQubitOscillationFrequency**: Idle qubit purity oscilation frequency

#### Gate Execution Quality Metrics
- **CliffordRB**: Clifford Randomized Benchmarking for average gate error
- **InterleavedRB**: Interleaved Randomized Benchmarking for specific gate fidelity
- **OverUnderRotationAngle**: Over/under rotation angle characterization
- **CycleBenchmarking**: Cycle benchmarking for composite process fidelity
- **GateSetTomography**: GST based process fidelity and SPAM fidelity 


#### Circuit Execution Quality Metrics
- **QuantumVolumeFixedQubits**: Quantum volume measurement for fixed qubit count
- **MirroredCircuits**: Mirrored Circuits average circuit polarization
- **UpperBoundOnVD**: Upper bound on the variation distance

#### Well-Studied Task Execution Quality Metrics
- **QFT**: Quantum Fourier Transform benchmark
- **VQE1DFermiHubbard**: Variational quantum eigensolver (VQE) metric
- **Simulation1DFermiHubbard**: Fermi-Hubbard model simulation (FHMS) metric
- **QScoreSingleInstance**: QScore application metric for fixed qubit count


## Hardware runs

In general, there are a few considerations that must be made when running on hardware.

1. Different quantum computers may have different native gates. This means that whilst the benchmarking circuits are designed to generate circuits using a specific gate set, they may not be able to be executed on some hardware without compilation to that specific set of native gates.
2. Often, changing parameters in a metric can change the number of circuits required to run as part of the metric. This will affect the run time and also change the costs associated with running the benchmark. In our CircuitSubmitter interface, we allow for the estimation of costs of submitting circuits by using pricing set by the backend.
3. Whilst running on an emulator means circuits can be executed instantly, often when running on hardware, you must input your circuits in a queue and wait for execution. This asynchronous execution means that the circuit submitter must query the hardware at a set frequency to check if the circuits have been submitted, and then retrieve the results.
4. Often in emulator runs, all-to-all connectivity of qubits is assumed. However, when running on hardware, this is not always the case. In addition to compiling to the native gates of the hardware, the circuits must be compiled such that they also respect the connectivity of multi-qubit gates available on the device.


## Contributing

We welcome contributions from the quantum computing community! Whether you're fixing bugs, adding new metrics, improving documentation, or suggesting features, your input helps advance quantum computing benchmarking for everyone.

1. **Read the Guidelines**: Review our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md)
2. **Set Up Development Environment**: Follow the setup instructions in [CONTRIBUTING.md](CONTRIBUTING.md)

## License

QCMet is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text and [NOTICE](NOTICE) for copyright information.

```
Copyright information can be found in NOTICE
Licensed under the Apache License, Version 2.0
```


## Citation

If you use QCMet in your research, please cite the accompanying article:


```bibtex
@article{qcmet2025,
  author = {D. Lall and A. Agarwal and W. Zhang and L. Lindoy and T. Lindström and S. Webster and S. Hall and N. Chancellor and P. Wallden and R. Garcia-Patron and E. Kashefi and V. Kendon and J. Pritchard and A. Rossi and A. Datta and T. Kapourniotis and K. Georgopoulos and I. Rungger},
  title = {A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software},
  journal = {arXiv preprint},
  year = {2025},
  eprint = {quant-ph/2502.06717},
  archivePrefix = {arXiv},
  url = {https://doi.org/10.48550/arXiv.2502.06717}
}
```
