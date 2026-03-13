"""QCMet module.

QCMet provides a collection of metrics and benchmarks for quantum
computers.
"""

from .benchmarks import (
    GST,
    QFT,
    T1,
    T2,
    VQE,
    BenchmarkCollection,
    CliffordRB,
    CycleBenchmarking,
    HamiltonianSimulation,
    IdleQubitOscillationFrequency,
    InterleavedRB,
    MirroredCircuits,
    OverUnderRotationAngle,
    QScore,
    QScoreSingleInstance,
    QuantumVolume,
    QuantumVolumeFixedQubits,
    SequentialBenchmark,
    Simulation1DFermiHubbard,
    UpperBoundOnVD,
    VQE1DFermiHubbard,
)
from .core import FileManager
from .devices import (
    AerSimulator,
    IdealSimulator,
    NoisySimulator,
    QiskitDevice,
)

__all__ = [
    "FileManager",
    "QuantumVolumeFixedQubits",
    "DummySimulator",
    "QFT",
    "GST",
    "HamiltonianSimulation",
    "Simulation1DFermiHubbard",
    "VQE",
    "VQE1DFermiHubbard",
    "QScoreSingleInstance",
    "CliffordRB",
    "CycleBenchmarking",
    "OverUnderRotationAngle",
    "FileManager",
    "QiskitDevice",
    "NoisySimulator",
    "IdealSimulator",
    "AerSimulator",
    "T1",
    "T2",
    "InterleavedRB",
    "MirroredCircuits",
    "IdleQubitOscillationFrequency",
    "UpperBoundOnVD",
    "BenchmarkCollection",
    "SequentialBenchmark",
    "QScore",
    "QuantumVolume",
]

__version__ = "1.0.0"
