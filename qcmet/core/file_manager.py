"""file_manager.py.

This module defines the FileManager class, which provides structured and
consistent file and directory management for saving benchmark data,
results, plots, configurations, and logs in the QCMet framework.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import qiskit.qasm3 as qasm3
from qiskit import QuantumCircuit
from qiskit.circuit import Gate


@dataclass
class FileManager:
    """Handles benchmark data saving and directory structure.

    This class creates a consistent directory layout for each
    benchmark run and provides methods to save JSON data, plots, and other
    intermediate results.

    The folders are set up in the following structure:

        base_path/
            └── benchmark_name/
                ├── results/
                │       └── plots/
                ├── data/
                ├── logs/
                ├── intermediate/
                └── config/


    Attributes:
        benchmark_name (str): Name of the benchmark.
        base_path (str | Path): Base directory where results will be saved.
        create_timestamp_folder (bool): Whether to create a timestamped subfolder.
        run_id (str): Unique identifier for the run (default is current timestamp).
        log_level (int): Logging level.
        save_circuits (bool): Whether to save quantum circuits.
        save_plots (bool): Whether to save plots.
        save_intermediate (bool): Whether to save intermediate data.

    """

    benchmark_name: str
    base_path: str | Path
    create_timestamp_folder: bool = True
    run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_level: int = logging.INFO
    save_circuits: bool = True
    save_plots: bool = True
    save_intermediate: bool = True

    def __post_init__(self):
        """Initialize the run path and create the directory structure."""
        self.base_path = Path(self.base_path)
        if self.create_timestamp_folder:
            self.run_path = self.base_path / f"{self.benchmark_name}_{self.run_id}"
        else:
            self.run_path = self.base_path / self.benchmark_name

        self._create_dir_structure()

    def _create_dir_structure(self):
        """Create a directory structure for storing benchmark outputs."""
        print(f"Creating dir structure at {self.run_path}")
        directories = [
            self.run_path,
            self.run_path / "results",
            self.run_path / "data",
            self.run_path / "results" / "plots",
            self.run_path / "config",
            self.run_path / "logs",
            self.run_path / "intermediate",
        ]

        for dir in directories:
            dir.mkdir(parents=True, exist_ok=True)

    def get_results_path(self) -> Path:
        """Get the path to the results directory.

        Returns:
            Path: Path to the results directory.

        """
        return self.run_path / "results"

    def get_data_path(self) -> Path:
        """Get the path to the data directory.

        Returns:
            Path: Path to the data directory.

        """
        return self.run_path / "data"

    def get_plots_path(self) -> Path:
        """Get the path to the plots directory.

        Returns:
            Path: Path to the plots directory.

        """
        return self.run_path / "results" / "plots"

    def get_config_path(self) -> Path:
        """Get the path to the config directory.

        Returns:
            Path: Path to the config directory.

        """
        return self.run_path / "config"

    def get_logs_path(self) -> Path:
        """Get the path to the logs directory.

        Returns:
            Path: Path to the logs directory.

        """
        return self.run_path / "logs"

    def get_intermediate_path(self) -> Path:
        """Get the path to the intermediate directory.

        Returns:
            Path: Path to the intermediate directory.

        """
        return self.run_path / "intermediate"

    def save_json(
        self, data: Dict[str, Any], filename: str, subfolder: str = "results"
    ) -> Path:
        """Save a dictionary as a JSON file in the specified subfolder.

        Args:
            data (Dict[str, Any]): Data to be saved.
            filename (str): Name of the JSON file (without extension).
            subfolder (str): Subfolder to save the file in.

        Returns:
            Path: Path to the saved JSON file.

        """
        if subfolder == "results":
            path = self.get_results_path()
        elif subfolder == "data":
            path = self.get_data_path()
        elif subfolder == "plots":
            path = self.get_plots_path()
        elif subfolder == "config":
            path = self.get_config_path()
        elif subfolder == "logs":
            path = self.get_logs_path()
        elif subfolder == "intermediate":
            path = self.get_intermediate_path()
        else:
            path = self.run_path / subfolder
            path.mkdir(exist_ok=True)
        filepath = path / f"{filename}.json"
        with open(filepath, "w") as f:
            json.dump(self._make_json_serializable(data), f, indent=4)
        return filepath

    def _make_json_serializable(self, obj):
        """Convert objects to a JSON-serializable format.

        Args:
            obj (Any): Object to convert.

        Returns:
            Any: JSON-serializable version of the object.

        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, Gate):
            return {
                "gate_name": obj.name,
                "gate_matrix": self._make_json_serializable(obj.to_matrix().tolist()),
            }
        elif isinstance(obj, QuantumCircuit):
            return qasm3.dumps(obj)
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def save_plot(self, figure, filename: str, format: str = "png") -> Path:
        """Save a matplotlib figure to the plots directory.

        Args:
            figure (matplotlib.figure.Figure): The figure to save.
            filename (str): Name of the file (without extension).
            format (str): File format (e.g., 'png', 'pdf').

        Returns:
            Path: Path to the saved plot file.

        """
        filepath = self.get_plots_path() / f"{filename}.{format}"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        return filepath
