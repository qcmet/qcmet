"""QiskitDevice subclass for quantum circuit benchmarking.

This module defines a device subclass 'QiskitDevice' based upon the 'BaseDevice' interface,
for devices operating with Qiskit.

Classes:
    QiskitDevice - a device subclass for executing and analyzing quantum circuits.
"""

from qcmet.devices.base_device import BaseDevice


class QiskitDevice(BaseDevice):
    """QiskitDevice implementation for quantum benchmarks.

    This subclass defines methods for qiskit-based devices. It supports bitstring
    reversal to match big-endian interpretation of measurement results.
    """

    def __init__(self, name="qiskit_device", basis_gates=None):
        """Initialize the Aer simulator device.

        Args:
            name (str): Subclass name.
            basis_gates (list[str], optional): Restrict transpiled circuits to specific gate set.

        """
        super().__init__(name)

    def reverse_bitstrings(self, counts_dict):
        """Reverse bitstrings in a measurement count dictionary.

        This converts between little and big endian.

        Args:
            counts_dict (dict): Original measurement counts with bitstrings as keys.

        Returns:
            dict: A new dictionary with bitstrings reversed.

        """
        big_endian_counts = {}
        for key, val in counts_dict.items():
            big_endian_counts[key[::-1]] = val
        return big_endian_counts
