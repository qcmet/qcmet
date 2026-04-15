"""base_device.py.

This module defines the abstract base class `BaseDevice` for quantum devices
within the QCMet framework. It establishes the interface that all quantum
device backends must implement to be compatible with BaseBenchmark.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDevice(ABC):
    """Abstract base class for quantum devices in the QCMet framework.

    This class defines the required interface for any quantum device or simulator
    that is used to execute quantum circuits and retrieve device-specific
    properties.

    Attributes:
        name (str): The name of the device.
        properties (Dict[str, Any]): A dictionary to store device-specific
            metadata or configuration.

    """

    def __init__(self, name: str):
        """Initialize the BaseDevice with a name.

        Args:
            name (str): The name of the quantum device.

        """
        self.name = name
        self.properties: Dict[str, Any] = {}

    # TODO do we define a set type for circuit/and return values
    @abstractmethod
    def _run(self, circuit, num_shots: int = 1024):
        """Execute one or more quantum circuits on the device.

        If a single circuit is passed, a single counts dictionary should be returned.
        If a list of circuits is passed, a list of counts dictionaries
        should be returned.

        Args:
            circuit (QuantumCircuit | List[QuantumCircuit]): The circuit(s) to execute.
            num_shots (int, optional): Number of measurement shots. Defaults to 1024.

        Returns:
            Dict[str, int] | List[Dict[str, int]]: Measurement outcomes (counts).

        """
        pass

    def run(self, circuits, num_shots: int = 1024, max_circs_per_job=None):
        """Execute one or more quantum circuits on the device.

        If a single circuit is passed, a single counts dictionary should be returned.
        If a list of circuits is passed, a list of counts dictionaries
        should be returned.

        Args:
            circuits (QuantumCircuit | List[QuantumCircuit]): The circuit(s) to execute.
            num_shots (int, optional): Number of measurement shots. Defaults to 1024.
            max_circs_per_job (int, optional): Maximum number of circuits to be submitted per job. This is for when
            benchmark requires more circuits than hardware can run in one job. Defaults to None.

        Returns:
            Dict[str, int] | List[Dict[str, int]]: Measurement outcomes (counts).
 
        """
        if max_circs_per_job:
            circ_jobs = [
                circuits[i : i + max_circs_per_job]
                for i in range(0, len(circuits), max_circs_per_job)
            ]
            counts = []
            for circs in circ_jobs:
                job_counts = self._run(circs, num_shots)
                if isinstance(job_counts, dict):
                    job_counts = [job_counts]
                counts.extend(job_counts)

        else:
            counts = self._run(circuits, num_shots)

        return counts
