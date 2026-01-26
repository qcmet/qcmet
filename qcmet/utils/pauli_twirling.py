"""Insert random Pauli twirls around selected two-qubit gates in a Qiskit DAG.

This transformation pass replaces each matched two-qubit gate with
``P_left -> gate -> P_right``, where the Pauli pair is chosen uniformly at random
from a precomputed set that preserves the unitary up to global phase.

This implementation is adapted from 
https://quantum.cloud.ibm.com/docs/en/guides/custom-transpiler-pass
"""

from typing import Iterable, Optional

import numpy as np
from qiskit.circuit import Gate, QuantumRegister
from qiskit.circuit.library import CXGate, CZGate, ECRGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Operator, pauli_basis
from qiskit.transpiler.basepasses import TransformationPass


class PauliTwirl(TransformationPass):
    """Randomly insert Pauli twirls around chosen two-qubit gates.

    For each targeted gate occurrence, the pass inserts a left and right Pauli so that
    the overall action is equivalent (up to global phase). By default, CX, ECR, and CZ
    are twirled.

    Attributes:
        gates_to_twirl: Iterable of two-qubit gate instances whose base classes define
            which gates are twirled.
        twirl_set: Mapping from gate name to valid Pauli pairs on two qubits.

    """

    def __init__(
        self,
        gates_to_twirl: Optional[Iterable[Gate]] = None,
        seed: Optional[int | np.random.Generator] = None,
    ):
        """Initialise the pass.

        Args:
            gates_to_twirl (optional, Iterable[Gate]): Gates to twirl. If ``None``, twirls CX, ECR, and CZ.
            seed (optional, int): Seeding random Pauli twirl pair selection. Defaults to None.

        """
        if gates_to_twirl is None:
            gates_to_twirl = [CXGate(), ECRGate(), CZGate()]
        self.gates_to_twirl = gates_to_twirl
        self.build_twirl_set()
        if isinstance(seed, int):
            self.rng = np.random.default_rng(seed)
        elif isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng()
        super().__init__()

    def build_twirl_set(self):
        """Precompute valid two-qubit Pauli pairs for each target gate.

        A pair ``(P_left, P_right)`` is kept if
        ``Operator(P_left) @ Operator(gate)`` is equivalent to
        ``Operator(gate) @ Operator(P_right)`` up to global phase.
        """
        self.twirl_set = {}

        # iterate through gates to be twirled
        for twirl_gate in self.gates_to_twirl:
            twirl_list = []

            # iterate through Paulis on left of gate to twirl
            for pauli_left in pauli_basis(2):
                # iterate through Paulis on right of gate to twirl
                for pauli_right in pauli_basis(2):
                    # save pairs that produce identical operation as gate to twirl
                    if (Operator(pauli_left) @ Operator(twirl_gate)).equiv(
                        Operator(twirl_gate) @ pauli_right
                    ):
                        twirl_list.append((pauli_left, pauli_right))

            self.twirl_set[twirl_gate.name] = twirl_list

    def run(
        self,
        dag: DAGCircuit,
    ) -> DAGCircuit:
        """Insert Pauli twirls around matching two-qubit gate nodes.

        For each matched node, replace it by ``P_left -> gate -> P_right``
        with a uniformly random Pauli pair from ``self.twirl_set``. If a seed
        is set on initialization of class, then it will be used here.
        
        Args:
            dag: Input DAGCircuit.

        Returns:
            The modified DAGCircuit.

        """
        # collect all nodes in DAG and proceed if it is to be twirled
        twirling_gate_classes = tuple(gate.base_class for gate in self.gates_to_twirl)
        for node in dag.op_nodes():
            if not isinstance(node.op, twirling_gate_classes):
                continue

            # random integer to select Pauli twirl pair
            pauli_index = self.rng.integers(0, len(self.twirl_set[node.op.name]))
            twirl_pair = self.twirl_set[node.op.name][pauli_index]

            # instantiate mini_dag and attach quantum register
            mini_dag = DAGCircuit()
            register = QuantumRegister(2)
            mini_dag.add_qreg(register)

            # apply left Pauli, gate to twirl, and right Pauli to empty mini-DAG
            mini_dag.apply_operation_back(
                twirl_pair[0].to_instruction(), [register[0], register[1]]
            )
            mini_dag.apply_operation_back(node.op, [register[0], register[1]])
            mini_dag.apply_operation_back(
                twirl_pair[1].to_instruction(), [register[0], register[1]]
            )

            # substitute gate to twirl node with twirling mini-DAG
            dag.substitute_node_with_dag(node, mini_dag)

        return dag
