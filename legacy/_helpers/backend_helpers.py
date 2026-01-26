from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from braket.aws import AwsDevice
from braket.circuits import gates
from qiskit_aer import AerSimulator
from qiskit_braket_provider import (
    AWSBraketProvider,
    BraketLocalBackend,
    AWSBraketBackend,
)
from qiskit_braket_provider.providers.adapter import *
from _helpers.noisy_simulator_wrappers import *
from _helpers.noise_model import custom_noise_model

class AwsBackendHelper(ABC):
    def __init__(self) -> None:
        self.device = None

    def get_device_calibration(self) -> str:
        return self.device.properties.json()

    @abstractmethod
    def get_qiskit_backend(self) -> Union[BraketLocalBackend, AWSBraketBackend]:
        pass

    @abstractmethod
    def get_basis_gates(self) -> list[str]:
        pass

    @abstractmethod
    def get_costs(self) -> Tuple[float]:
        """Get cost_per_circuit and cost_per_shot"""
        pass


class IdealSimulatorHelper(AwsBackendHelper):
    def __init__(self) -> None:
        super().__init__()
        from braket.devices import LocalSimulator

        self.device = LocalSimulator()

    def get_device_calibration(self):
        return "Local simulator with no noise"

    def get_qiskit_backend(self):
        return BraketLocalBackend()

    def get_basis_gates(self):
        return ["rx", "ry", "rz", "h", "cx", "id"]

    def get_costs(self) -> Tuple[float]:
        return 0, 0


class NoisySimulatorHelper(AwsBackendHelper):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if 'noise_model' in kwargs:
            self.noise_model = kwargs.get('noise_model')
            self.device = NoisySimWrapper(noise_model = self.noise_model)
        else:
            self.device = NoisySimWrapper()
        self.name = "noisy_sim"

    def get_device_calibration(self):
        return "Local simulator with two-qubit noise model with perfect shots:\n" + str(
            self.device.noise_model
        )

    def get_qiskit_backend(self):
        return BraketLocalBackend()

    def get_basis_gates(self):
        if self.noise_model:
            return self.noise_model.basis_gates
        else:
            return ["sx", "rz", "cx", "id"]

    def get_costs(self) -> Tuple[float]:
        return 0, 0
    

class NoisySimulatorHelperWithShots(AwsBackendHelper):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if 'noise_model' in kwargs:
            self.noise_model = kwargs.get('noise_model')
            self.device = NoisySimWrapperWithShots(noise_model = self.noise_model)
        else:
            self.device = NoisySimWrapperWithShots()
        self.name = "noisy_sim_with_shots"

    def get_device_calibration(self):
        return "Local simulator with noise model with shots noise:\n" + str(
            self.device.noise_model
        )

    def get_qiskit_backend(self):
        return BraketLocalBackend()

    def get_basis_gates(self):
        if self.noise_model:
            return self.noise_model.basis_gates
        else:
            return ["sx", "rz", "cx", "id"]

    def get_costs(self) -> Tuple[float]:
        return 0, 0


    def get_qiskit_backend(self):
        return BraketLocalBackend()

    def get_basis_gates(self):
        if self.noise_model:
            return self.noise_model.basis_gates
        else:
            return ["sx", "rz", "cx", "id"]

    def get_costs(self) -> Tuple[float]:
        return 0, 0


class DensityMatrixIdealSimulatorHelper(AwsBackendHelper):
    def __init__(self) -> None:
        super().__init__()
        self.device = NoiselessDensityMatrixSimWrapper()
        self.name = "noiseless_sim"

    def get_device_calibration(self):
        return "Local noiseless simulator with perfect shots:\n"

    def get_qiskit_backend(self):
        return BraketLocalBackend()

    def get_basis_gates(self):
        return ["rx", "ry", "rz", "cx", "id"]

    def get_costs(self) -> Tuple[float]:
        return 0, 0


class OQCLucyHelper(AwsBackendHelper):
    def __init__(self) -> None:
        super().__init__()
        self.device = AwsDevice("arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy")

    def get_qiskit_backend(self) -> AWSBraketBackend:
        return AWSBraketProvider().get_backend("Lucy")

    def get_basis_gates(self) -> list[str]:
        return ["ecr", "i", "rz", "sx", "x"]

    def get_costs(self) -> Tuple[float]:
        return 0.3, 0.00035


class OQCDirectHelper(AwsBackendHelper):
    def __init__(self) -> None:
        from _helpers.oqc_direct_wrappers import OQCClientWrapper

        super().__init__()
        self.device = OQCClientWrapper()

    def get_device_calibration(self) -> str:
        return AwsDevice(
            "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy"
        ).properties.json()

    def get_qiskit_backend(self):
        return BraketLocalBackend()

    def get_basis_gates(self) -> list[str]:
        return ["ecr", "i", "rz", "sx", "x"]

    def get_costs(self) -> Tuple[float]:
        return 0, 0


class IonQHarmonyHelper(AwsBackendHelper):
    def __init__(self) -> None:
        super().__init__()
        self.device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")

    def get_qiskit_backend(self) -> AWSBraketBackend:
        return AWSBraketProvider().get_backend("Harmony")

    def get_basis_gates(self) -> list[str]:
        # Add equivalences for transpiler
        qiskit_gate_names_to_braket_gates["x"] = lambda: [gates.GPi(0)]
        qiskit_gate_names_to_braket_gates["y"] = lambda: [gates.GPi(np.pi / 2)]
        qiskit_gate_names_to_braket_gates["sx"] = lambda: [gates.GPi2(np.pi)]
        qiskit_gate_names_to_braket_gates["h"] = lambda: [
            gates.GPi2(np.pi / 2),
            gates.GPi(0),
        ]
        qiskit_gate_names_to_braket_gates["rxx"] = lambda angle: [gates.MS(0, 0, angle)]
        return ["x", "y", "sx", "h", "rxx"]

    def get_costs(self) -> Tuple[float]:
        return 0.1, 0.01


class IonQAriaHelper(AwsBackendHelper):
    def __init__(self) -> None:
        super().__init__()
        self.device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")

    def get_qiskit_backend(self) -> AWSBraketBackend:
        return AWSBraketProvider().get_backend("Harmony")

    def get_basis_gates(self) -> list[str]:
        # Add equivalences for transpiler
        qiskit_gate_names_to_braket_gates["x"] = lambda: [gates.GPi(0)]
        qiskit_gate_names_to_braket_gates["y"] = lambda: [gates.GPi(np.pi / 2)]
        qiskit_gate_names_to_braket_gates["sx"] = lambda: [gates.GPi2(np.pi)]
        qiskit_gate_names_to_braket_gates["h"] = lambda: [
            gates.GPi2(np.pi / 2),
            gates.GPi(0),
        ]
        qiskit_gate_names_to_braket_gates["rxx"] = lambda angle: [gates.MS(0, 0, angle)]
        return ["x", "y", "sx", "h", "rxx"]

    def get_costs(self) -> Tuple[float]:
        return 0.1, 0.03


def get_backend_helper(name: str) -> AwsBackendHelper:
    if name == "simulator":
        return IdealSimulatorHelper()
    elif name == "noisy_sim":
        return NoisySimulatorHelper()
    elif name == "noisy_sim_with_shots":
        return NoisySimulatorHelperWithShots()
    elif name == "noiseless_sim":
        return DensityMatrixIdealSimulatorHelper()
    elif name == "Lucy":
        raise ValueError(f"{name} Device no longer supported")
        # return OQCLucyHelper()
    elif name == "OQCDirect":
        raise ValueError(f"{name} Device no longer supported")
        # return OQCDirectHelper()
    elif name == "Harmony":
        raise ValueError(f"{name} Device no longer supported")
        # return IonQHarmonyHelper()
    elif name == "Aria":
        return IonQAriaHelper()
    else:
        raise ValueError(f"Unsupported device name {name}")


if __name__ == "__main__":
    print(get_backend_helper("simulator").get_costs())