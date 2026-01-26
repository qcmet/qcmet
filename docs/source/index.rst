QCMet - Quantum Computing Metrics and Benchmarks
================================================

.. raw:: html

   <p class="lead">
   A comprehensive collection of metrics and benchmarks for quantum computers.
   </p>

.. note::
   QCMet is the software accompanying the article `"A Review and Collection of Metrics
   and Benchmarks for Quantum Computers: definitions, methodologies and software"
   <https://doi.org/10.48550/arXiv.2502.06717>`_.

Why QCMet?
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Python
      :text-align: center

      QCMet is written in Python for readability and ease of use.

   .. grid-item-card:: Comprehensive
      :text-align: center

      Covers a wide range of quantum computing metrics and benchmarks from qubit quality to full application-level benchmarks.

   .. grid-item-card:: Hardware Agnostic
      :text-align: center

      Works with multiple quantum computing platforms including simulators and real quantum hardware.

   .. grid-item-card:: Free and Open-Source
      :text-align: center

      Licensed under Apache-2.0, QCMet is free to use and modify.


Key Features
------------

QCMet provides implementations of various quantum computing metrics organized into categories:

- **Qubit Quality Metrics**: T1, T2, idle qubit oscillation frequency
- **Gate Execution Quality Metrics**: Randomized benchmarking (Clifford RB, Interleaved RB), over/under-rotation analysis, cycle benchmarking, gate set tomography (vie pyGSTi)
- **Circuit Execution Quality Metrics**: Quantum Volume, mirrored circuits, upper bound on the variation distance
- **Well-Studied Task Execution Quality Metrics**: QFT, VQE, Hamiltonian simulation, QScore

The software is designed with a device interface that allows evaluation of metrics using:

- Local simulators (ideal and noisy simulations via Qiskit Aer)
- Real quantum computers via compatible device backends
- Custom device implementations


Getting Started
---------------

Installation
^^^^^^^^^^^^
Download the code base and install QCMet via pip:

.. code-block:: bash

   pip install -e .


Quick Example
^^^^^^^^^^^^^

.. code-block:: python

   from qcmet import T1
   from qcmet.devices import IdealSimulator
   import numpy as np

   # Initialize simulator
   device = IdealSimulator()

   # Create T1 benchmark
   t1 = T1(num_idle_gates_per_circ=np.arange(1, 2000, 200))

   # The following generates the circuits, runs them on the device and analyzes the benchmark
   results = t1(device, num_shots=1024)
   print(f"T1 Results: {results}")


.. toctree::
   :maxdepth: 1
   :hidden:

   tutorials/01_installation
   tutorials/02_quickstart
   user_guide
   api/index


Citation
--------

If you use QCMet in your work, please cite:

   D. Lall, A Agarwal, W. Zhang, L. Lindoy, T. Lindström, S. Webster, S. Hall, N. Chancellor, P. Wallden, R. Garcia-Patron, E. Kashefi, V. Kendon, J. Pritchard, A. Rossi, A. Datta, T. Kapourniotis, K. Georgopoulos, I. Rungger, *A Review and Collection of Metrics and Benchmarks for Quantum Computers: definitions, methodologies and software*, arXiv:2502.06717 (2025).
