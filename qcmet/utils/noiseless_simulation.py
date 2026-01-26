"""noiseless_simulation.py.

This module provides the utlility functions for noiseless simulation required for benchmarks.
"""

from qiskit.quantum_info import Statevector


def compute_ideal_outputs(qc):
    """Compute the ideal probabilities of the quantum circuit for all possible reversed (big-endian) bitstrings.

    Args:
        qc (QuantumCircuit): The square quantum volume circuit.

    Returns:
        Dict: Bitstring outputs and their corresponding noiseless probabilities.

    """
    sv = final_statevector(qc)

    probs = {}
    for i, amplitude in enumerate(sv):
        bitstring = format(i, f"0{sv.num_qubits}b")
        probs[bitstring] = abs(amplitude) ** 2

    sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
    ideal_outputs_unreversed = {str(k): float(v) for k, v in sorted_probs.items()}

    ideal_outputs = {}
    for key, val in ideal_outputs_unreversed.items():
        ideal_outputs[key[::-1]] = val

    return ideal_outputs


def final_statevector(qc):
    """Remove the final measurement of a quantum circuit and return the statevector of the circuit.

    Args:
        qc (QuantumCircuit): The square quantum volume circuit.


    Returns:
        Statevector: The statevector of the quantum circuit.

    """
    qc.remove_final_measurements()

    return Statevector.from_instruction(qc)
