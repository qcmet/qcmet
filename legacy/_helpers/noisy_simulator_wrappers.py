import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit.library import RZXGate
from qiskit.circuit.measure import Measure
from qiskit_aer import AerSimulator, AerJob
from qiskit_aer.noise import NoiseModel
import numpy as np
from _helpers.noise_model import custom_noise_model


class QiskitTaskResultWrapper:
    def __init__(self, result: AerJob, shots: int) -> None:
        result = result.result()
        num_qubits = result.results[0].header.n_qubits
        output_state = np.abs(
            np.asarray(result.data()["before_measurement"], dtype=np.complex128)
        )
        eps = np.finfo(float).eps
        output_state[np.abs(output_state) < eps] = 0
        output_state = np.rint(output_state * shots)

        counts = {
            format(b, f"0{num_qubits}b"): int(v)
            for b, v in enumerate(np.diag(output_state))
        }

        # Adjust the highest valued count to align with the shots
        highest_count_key = max(counts, key=counts.get)
        sum_counts = sum(counts.values())
        counts[highest_count_key] -= sum_counts - shots

        # Convert qisit order to braket order, so this is consistent with other wrappers
        counts = {k[::-1]: v for k, v in counts.items()}
        self.measurement_counts = counts


class QiskitTaskResultWrapperWithShots:
    def __init__(self, result: AerJob, shots: int) -> None:
        result = result.result()
        num_qubits = result.results[0].header.n_qubits
        # print(dir(result))
        self.measurement_counts = result.get_counts(0)


class QiskitTaskWrapper:
    def __init__(self, task: AerJob, shots: int, shot_noise: bool) -> None:
        self.task = task
        self.id = task.job_id()
        self.shots = shots
        self.shot_noise = shot_noise

    def result(self):
        if self.shot_noise:
            return QiskitTaskResultWrapperWithShots(self.task, self.shots)
        else:
            return QiskitTaskResultWrapper(self.task, self.shots)

    def state(self):
        return "COMPLETED"


class QiskitTaskBatchWrapper:
    def __init__(self, tasks: list[QiskitTaskWrapper]) -> None:
        self.tasks = tasks


class SimWrapper:
    def __init__(
        self, backend: Backend = None, noise_model: NoiseModel = None, shot_noise=False
    ):
        self.shot_noise = shot_noise
        if backend is None:
            self.backend = AerSimulator
        else:
            self.backend = backend
        if noise_model is not None:
            self.noise_model = noise_model
            self.sim = self.backend(
                method="density_matrix", noise_model=self.noise_model
            )
        else:
            self.noise_model = None
            self.sim = self.backend(method="density_matrix")

    def _remove_measurement_and_add_dm_save(self, circ):
        index_to_delete = []
        for index, instruction in enumerate(circ.data):
            if isinstance(instruction.operation, Measure):
                index_to_delete.append(index)
        for index in reversed(index_to_delete):
            del circ.data[index]
        circ.save_density_matrix(label="before_measurement")
        return circ

    def run_batch(
        self, circuits: list[QuantumCircuit], shots=1000, max_parallel=100, **kwargs
    ):
        self.shots = shots
        if self.shot_noise:
            task_batch = QiskitTaskBatchWrapper(
                [
                    QiskitTaskWrapper(
                        self.sim.run(c.reverse_bits(), shots=shots),
                        shots=shots,
                        shot_noise=True,
                    )
                    for c in circuits
                ]
            )
            return task_batch
        else:
            circuits2 = []
            for c in circuits:
                c = self._remove_measurement_and_add_dm_save(c)
                circuits2.append(c)
            task_batch = QiskitTaskBatchWrapper(
                [
                    QiskitTaskWrapper(
                        self.sim.run(c, shots=shots), shots=shots, shot_noise=False
                    )
                    for c in circuits2
                ]
            )
            return task_batch



class NoisySimWrapper(SimWrapper):
    def __init__(self, backend: Backend = None, noise_model: NoiseModel = None):

        self.noise_model = noise_model
        self.backend = backend

        if self.backend is None:
            self.backend = AerSimulator

        if self.noise_model is None:
            self.noise_model = custom_noise_model()

        super().__init__(backend=self.backend, noise_model=self.noise_model)

    def set_noise_model(self, noise_model):
        super().__init__(self.backend, noise_model=noise_model)


class NoisySimWrapperWithShots(SimWrapper):
    def __init__(self, backend: Backend = None, noise_model: NoiseModel = None):

        self.noise_model = custom_noise_model()
        self.backend = backend

        if self.backend is None:
            self.backend = AerSimulator

        if noise_model is not None:
            self.noise_model = noise_model

        super().__init__(backend=self.backend, noise_model=self.noise_model,  shot_noise=True)

    def set_noise_model(self, noise_model):
        super().__init__(self.backend, noise_model=noise_model, shot_noise=True)


class NoiselessDensityMatrixSimWrapper(SimWrapper):
    def __init__(self):
        super().__init__()
