"""Upper bound on the variation distance metric.

This module implements a benchmark for the upper bound on the variation
distance (VD) between the probability distribution of the experimental outputs
of a noisy quantum circuit and its noiseless counterparts. This benchmark uses
the quantum accreditation protocol (AP). Here the benchmarking procedure
follows M4.4 from arxiv:2502.06717.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from pathlib import Path

    from qcmet.core import FileManager

import numpy as np
from qiskit import QuantumCircuit

from qcmet.benchmarks import BaseBenchmark


class UpperBoundOnVD(BaseBenchmark):
    """Implementation of the upper bound on the variation distance (VD) metric.

    This class generates a number of trap circuits with similar structure to that
    of a given target circuit, according to the quantum accreditation  protocol (AP).
    The trap circuits should output zero states in the noiseless case, thereby
    the number of non-zero outputs is used to compute the upper bound on the VD.
    """

    def __init__(
        self,
        target_circuit: QuantumCircuit,
        mu: float = 0.0001,
        eta: float = 0.9999,
        seed: int | None = None,
        save_path: str | Path | FileManager | None = None,
    ):
        """Initialize the upper bound on the VD benchmark.

        Args:
            target_circuit (QuantumCircuit): A target circuit to estimate the upper bound on VD for.
                The target circuit must follow the restriction such that it has alternating cycles
                of one-qubit and two-qubit gates, and the two-qubit gates must be CZ gates.
            mu (float, optional): The desired accuracy of the benchmark ∈ (0, 1). Defaults to 0.0001.
            eta (float, optional): The desired confidence of the benchmark ∈ (0, 1). Defaults to 0.9999.
            seed (int, optional): Random seed to use for randomisations. Defaults to None.
            save_path (str | Path | FileManager | None, optional): Directory path to save results. Defaults to None.

        """
        super().__init__(
            "UpperBoundOnVD", target_circuit.num_qubits, save_path=save_path
        )
        self.config["target_circuit"] = target_circuit
        self.config["mu"] = mu
        self.config["eta"] = eta
        self.config["num_trap_circuits"] = int(
            np.ceil(2 * np.log((2 / (1 - eta)) / mu**2))
        )

        if seed is None:
            seed = np.random.randint(100000000)
        self.config["seed"] = seed
        self._rng = np.random.default_rng(seed)

        # self._target_circuit_gate_existence_in_cycles: List[dict[str, Any]] | None = None

    def parse_target_circuit(self, target_circuit: QuantumCircuit):
        """Parse the target circuit's structure.

        This method breaks down the circuit structure in terms of alternating cycles of one- and
        two-qubit gates. Each cycle starts with one-qubit gate(s) and ends with (possibly parallel)
        two-qubit gate(s). Gates up to the next non-parallel two-qubit gate becomes the next cycle.
        A dictionary is generated for each cycle, which tells for each qubit in this cycle whether
        one-qubit gate(s) or a two-qubit gate is existent on this qubit. This information is later
        used for constructing trap circuits.

        Args:
            target_circuit (QuantumCircuit): The target circuit to parse.

        Returns:
            list: The list of circuit structure information, one element for each cycle.
            Each element is a dict:
            {
            "1q": [bool],
            "2q": [[int, int, CircuitInstruction]]
            }.
            The 1q list in dict has length equal to the number of qubits, which tells whether
            a one-qubit gate exists for each qubit. The 2q list has length equal to the number
            of parallel two-qubit gates in this cycle, and contains the qubit indices and
            the 2q gate itself.

        """
        circuit_data = target_circuit.data
        cycle_tracker = [0 for _ in range(target_circuit.num_qubits)]
        cycle_index_of_each_gate = []

        for gate in circuit_data:
            if gate.operation.name in ["measure", "barrier"]:
                cycle_index_of_each_gate.append(-1)
                continue
            num_qubits = gate.operation.num_qubits
            if num_qubits not in [1, 2]:
                raise ValueError(
                    f"The target circuit can only have one or two qubit gates. Wrong gate: {gate}."
                )

            if num_qubits == 1:
                cycle_index_of_each_gate.append(cycle_tracker[gate.qubits[0]._index])
            else:
                if gate.operation.name != "cz":
                    raise ValueError(
                        f"Two-qubit gates must be CZ gates. Wrong gate: {gate}."
                    )
                qubit_index0 = gate.qubits[0]._index
                qubit_index1 = gate.qubits[1]._index
                max_depth = max(
                    cycle_tracker[qubit_index0], cycle_tracker[qubit_index1]
                )
                cycle_index_of_each_gate.append(max_depth)
                cycle_tracker[qubit_index0] = max_depth + 1
                cycle_tracker[qubit_index1] = max_depth + 1

        self._target_circuit_gate_existence_in_cycles: List[dict[str, Any]] = [
            {"1q": [False for _ in range(target_circuit.num_qubits)], "2q": []}
            for _ in range(max(cycle_index_of_each_gate) + 1)
        ]

        for gate, cycle_index in zip(
            circuit_data, cycle_index_of_each_gate, strict=True
        ):
            if gate.operation.name in ["measure", "barrier"]:
                continue
            if gate.operation.num_qubits == 1:
                self._target_circuit_gate_existence_in_cycles[cycle_index][
                    f"{gate.operation.num_qubits}q"
                ][gate.qubits[0]._index] = True
            else:
                self._target_circuit_gate_existence_in_cycles[cycle_index][
                    f"{gate.operation.num_qubits}q"
                ].append([gate.qubits[0]._index, gate.qubits[1]._index, gate])

        return self._target_circuit_gate_existence_in_cycles

    @staticmethod
    def generate_example_target_circuit(
        num_qubits: int, cycles: int, seed: int | None = None
    ):
        """Randomly generate an example target circuit that satisfies the restriction.

        Args:
            num_qubits (int): The number of qubits.
            cycles (int): The number of cycles.
            seed (int, optional): Random seed. Defaults to None.

        Returns:
            QuantumCircuit: An example target circuit.

        """
        circuit = QuantumCircuit(num_qubits)
        gates = [circuit.h, circuit.x, circuit.y, circuit.z]
        rng = np.random.default_rng(seed)

        for _ in range(cycles):
            for qubit in np.sort(
                rng.choice(num_qubits, rng.integers(1, num_qubits + 1), replace=False)
            ):
                gates[rng.integers(len(gates))](qubit)
            for qubit1, qubit2 in rng.choice(
                num_qubits, rng.integers(1, num_qubits // 2 + 1) * 2, replace=False
            ).reshape((-1, 2)):
                circuit.cz(qubit1, qubit2)
            circuit.barrier()
        return circuit

    def _generate_circuits(self):
        """Generate trap circuits for the quantum AP to benchmark the upper bound on VD.

        The number of trap circuits is equal to ⌈2ln(2/(1−η))/μ^2⌉.
        Each trap circuit is generated in the following way:
            1. For each cycle in the target circuit:
                a. If the j-th cycle of CZ gates connects qubit i to qubit i′, randomly replace
                    the one-qubit gates on i and i' with [H, S] or [S, H]. Inverses of these
                    gates are added to the end of the cycle.
                b. If there's a one-qubit gate but not a two-qubit gate on a qubit in this cycle,
                    randomly replace it with H or S, and add the inverse to the end of the cycle.
                c. Add random Pauli gates right before the CZ gates. Add corresponding Pauli gates
                    right after the CZ gates so that together they are equivalent to CZ.
            2. Randomly decide whether to add a column of H gates to both the start and the
                end of the circuit.

        Returns:
            list: A list of QuantumCircuit objects, which should output zero states in the ideal case.

        """

        def _add_h_gates(flag, circuit):
            if flag:
                for i in range(circuit.num_qubits):
                    circuit.h(i)

        self.parse_target_circuit(self.config["target_circuit"])

        trap_circuits = []
        for _ in range(self.config["num_trap_circuits"]):
            trap_circuit = QuantumCircuit(self.config["target_circuit"].num_qubits)
            trap_circuit_replacer_gates = [trap_circuit.h, trap_circuit.s]
            trap_circuit_undo_gates = [trap_circuit.h, trap_circuit.sdg]
            trap_circuit_1paulis = [
                trap_circuit.id,
                trap_circuit.x,
                trap_circuit.y,
                trap_circuit.z,
            ]
            trap_circuit_2paulis = [
                [trap_circuit.id, trap_circuit.x],
                [trap_circuit.x, trap_circuit.id],
                [trap_circuit.x, trap_circuit.z],
                [trap_circuit.z, trap_circuit.x],
            ]
            trap_circuit_undo_2paulis = [
                [trap_circuit.z, trap_circuit.x],
                [trap_circuit.x, trap_circuit.z],
                [trap_circuit.x, trap_circuit.id],
                [trap_circuit.id, trap_circuit.x],
            ]

            # Randomly decide if we add additional H gates at the beginning and end
            should_add_h_gates = self._rng.integers(2) == 0
            _add_h_gates(should_add_h_gates, trap_circuit)

            # For each cycle
            for (
                gate_existence_in_each_cycle
            ) in self._target_circuit_gate_existence_in_cycles:
                undo_gates_list: List[Any] = []
                # For each single qubit gate in each cycle
                for qubit_index, oneq_exist in enumerate(
                    gate_existence_in_each_cycle["1q"]
                ):
                    oneq_gate_connects_twoq_gate: bool = any(
                        qubit_index in twoq_gate_indices
                        for twoq_gate_indices in gate_existence_in_each_cycle["2q"][:2]
                    )
                    # Add gates only if there is 1q gate on this qubit in this cycle but no 2q gate
                    if oneq_exist and not oneq_gate_connects_twoq_gate:
                        # Randomly replace the 1q gate with H or S
                        random_replace = self._rng.integers(2)
                        trap_circuit_replacer_gates[random_replace](qubit_index)

                        # Add random 1q Pauli
                        random_pauli = self._rng.integers(4)
                        trap_circuit_1paulis[random_pauli](qubit_index)
                        undo_gates_list.append(
                            [qubit_index, trap_circuit_1paulis[random_pauli]]
                        )

                        undo_gates_list.append(
                            [qubit_index, trap_circuit_undo_gates[random_replace]]
                        )

                # For each 2q gate, we'll add/replace 1q gates
                for qubit_index0, qubit_index1, gate in gate_existence_in_each_cycle[
                    "2q"
                ]:
                    # Randomly replace the 1q gates (or add if there is none) with [H and S] or [S and H]
                    random_replace = self._rng.integers(2)
                    trap_circuit_replacer_gates[random_replace](qubit_index0)
                    trap_circuit_replacer_gates[1 - random_replace](qubit_index1)

                    # Add random 2q Pauli
                    random_pauli = self._rng.integers(4)
                    trap_circuit_2paulis[random_pauli][0](qubit_index0)
                    trap_circuit_2paulis[random_pauli][1](qubit_index1)
                    trap_circuit.append(gate)
                    undo_gates_list.append(
                        [qubit_index0, trap_circuit_undo_2paulis[random_pauli][0]]
                    )
                    undo_gates_list.append(
                        [qubit_index1, trap_circuit_undo_2paulis[random_pauli][1]]
                    )

                    undo_gates_list.append(
                        [qubit_index0, trap_circuit_undo_gates[random_replace]]
                    )
                    undo_gates_list.append(
                        [qubit_index1, trap_circuit_undo_gates[1 - random_replace]]
                    )

                # Add the so-called undo gates
                for undo_index, undo_gate in undo_gates_list:
                    undo_gate(undo_index)

                # Add barrier in between cycles
                trap_circuit.barrier()

            # Add ending H gates
            _add_h_gates(should_add_h_gates, trap_circuit)

            trap_circuit.measure_all()

            trap_circuits.append(trap_circuit)

        return trap_circuits

    def _analyze(self):
        """Analyze measurement results for the upper bound on VD metric.

        Since all trap circuits should output zero states in the ideal case,
        the upper bound on VD is calculated by the number of non-zero outputs divided
        by the total number of measurement shots.

        Returns:
            dict: {
              "fails": int,
              "total": int,
              "upper_bound_on_vd": float
            }

        """
        fails = 0
        total = 0
        for counts_dict in self.experiment_data["circuit_measurements"]:
            for key, value in counts_dict.items():
                total += value
                if int(key) != 0:
                    fails += value
        b = 2 * fails / total
        result = {"fails": fails, "total": total, "upper_bound_on_vd": b}

        return result
