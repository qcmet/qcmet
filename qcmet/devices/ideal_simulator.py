"""Defines IdealSimulator device based on qiskit aer_simulator."""

from qcmet.devices import AerSimulator


class IdealSimulator(AerSimulator):
    """Noiseless AerSimulator."""

    def __init__(self):
        """Construct IdealSimulator."""
        super().__init__(noise_model= None, basis_gates=["u1", "u2", "u3", "cx"])