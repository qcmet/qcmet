from __future__ import annotations

from typing import Any, List, Tuple

from qiskit import QuantumCircuit


def _active_qubits(circuit: QuantumCircuit) -> set[int]:
    """Return the set of qubit indices touched by any instruction in the circuit."""
    active = set()
    for inst in circuit.data:
        active.update(circuit.find_bit(q).index for q in inst.qubits)
    return active


def _validate_disjoint_groups(circuit_groups: List[List[QuantumCircuit]]) -> None:
    """Check that different circuit groups act on disjoint qubits.

    For each group, the active qubits are taken as the union of qubits touched by
    all circuits in that group.
    """
    used_qubits = set()

    for i, circuits in enumerate(circuit_groups):
        group_active_qubits = set()
        for circuit in circuits:
            group_active_qubits.update(_active_qubits(circuit))

        overlap = used_qubits & group_active_qubits
        if overlap:
            raise ValueError(
                f"Circuit group {i} overlaps with a previous group on "
                f"qubits {sorted(overlap)}. Fusion requires disjoint qubits."
            )

        used_qubits |= group_active_qubits


def _tokenize_circuit_by_barriers(circuit: QuantumCircuit) -> List[Tuple[str, Any]]:
    """Split a circuit into alternating 'segment' and 'barrier' tokens.

    Consecutive barriers are merged into a single barrier token.
    """
    tokens = []
    current_segment = []

    i = 0
    while i < len(circuit.data):
        inst = circuit.data[i]

        if inst.operation.name == "barrier":
            if current_segment:
                tokens.append(("segment", current_segment))
                current_segment = []

            barrier_qubits = {circuit.find_bit(q).index for q in inst.qubits}
            i += 1

            while i < len(circuit.data) and circuit.data[i].operation.name == "barrier":
                next_inst = circuit.data[i]
                barrier_qubits.update(circuit.find_bit(q).index for q in next_inst.qubits)
                i += 1

            tokens.append(("barrier", sorted(barrier_qubits)))
        else:
            current_segment.append(inst)
            i += 1

    if current_segment:
        tokens.append(("segment", current_segment))

    return tokens


def _append_segment_to_fused(
    fused: QuantumCircuit,
    source_circuit: QuantumCircuit,
    instructions: List[Any],
    cindex_map: List[int],
) -> None:
    """Append one instruction segment into the fused circuit.

    Source-circuit qubits are assumed to already live on their final qubit indices.
    Classical bits are remapped into the group's reserved clbit slice.
    """
    for inst in instructions:
        qindices = [source_circuit.find_bit(q).index for q in inst.qubits]
        cindices = [source_circuit.find_bit(c).index for c in inst.clbits]

        fused_qargs = [fused.qubits[i] for i in qindices]
        fused_cargs = [fused.clbits[cindex_map[i]] for i in cindices]

        op = inst.operation.copy() if hasattr(inst.operation, "copy") else inst.operation
        fused.append(op, fused_qargs, fused_cargs)


def fuse_circuit_groups(
    circuit_groups: List[List[QuantumCircuit]],
    fuse_mode: str = "strict",
) -> Tuple[List[QuantumCircuit], List[List[int]]]:
    """Fuse multiple groups of circuits into larger parallel circuits.

    Assumptions:
    - Every input circuit already lives on its final qubit indices.
    - Different groups act on disjoint qubits.

    Args:
        circuit_groups (List[List[QuantumCircuit]]): Each group corresponds to the list of circuits of one benchmark.
        fuse_mode ("strict" or "min" or "pad"):
            How to handle groups of different lengths:
            - "strict": all groups must have the same number of circuits
            - "min": only fuse up to the shortest group length
            - "pad": fuse up to the longest group length; shorter groups simply
                     contribute nothing to later fused circuits

    Returns:
        fused_circuits (List[QuantumCircuit]):
            The fused execution circuits.
        clbit_layout (List[List[int]]):
            A list aligned with the input groups. clbit_layout[i] is the classical-bit
            slice reserved for group i inside each fused circuit.
    """
    normalized_groups = []
    lengths = []
    num_clbits_per_group = []
    max_num_qubits = 0

    for i, circuits in enumerate(circuit_groups):
        circuits = list(circuits)

        if not circuits:
            raise ValueError(f"circuit_groups[{i}] is empty")

        if not all(isinstance(circ, QuantumCircuit) for circ in circuits):
            raise ValueError(f"circuit_groups[{i}] contains non-QuantumCircuit entries")

        clbit_sizes = {circ.num_clbits for circ in circuits}
        if len(clbit_sizes) != 1:
            raise ValueError(
                f"Currently it's only supported that the number of classical bits is the same within a circuit group. "
                f"circuit_groups[{i}] has inconsistent clbits: {sorted(clbit_sizes)}."
            )

        normalized_groups.append(circuits)
        lengths.append(len(circuits))
        num_clbits_per_group.append(circuits[0].num_clbits)
        max_num_qubits = max(max_num_qubits, *(circ.num_qubits for circ in circuits))

    _validate_disjoint_groups(normalized_groups)

    if fuse_mode == "strict":
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Groups have different numbers of circuits: {lengths}. "
                "Use fuse_mode='min' or 'pad' if that is intended."
            )
        num_fused = lengths[0]
    elif fuse_mode == "min":
        num_fused = min(lengths)
    elif fuse_mode == "pad":
        num_fused = max(lengths)
    else:
        raise ValueError("fuse_mode must be one of: 'strict', 'min', 'pad'")

    clbit_layout = []
    clbit_offset = 0
    for num_clbits in num_clbits_per_group:
        clbits = list(range(clbit_offset, clbit_offset + num_clbits))
        clbit_layout.append(clbits)
        clbit_offset += num_clbits
    total_clbits = clbit_offset

    fused_circuits = []

    for fused_index in range(num_fused):
        fused = QuantumCircuit(max_num_qubits, total_clbits, name=f"fused_{fused_index}")
        streams = []

        for group_index, circuits in enumerate(normalized_groups):
            if fused_index >= len(circuits):
                continue

            circ = circuits[fused_index]
            tokens = _tokenize_circuit_by_barriers(circ)

            fused.global_phase += circ.global_phase

            streams.append(
                {
                    "circ": circ,
                    "tokens": tokens,
                    "pointer": 0,
                    "cindex_map": clbit_layout[group_index],
                }
            )

        while any(stream["pointer"] < len(stream["tokens"]) for stream in streams):
            progressed = False

            # Append all next non-barrier segments first
            for stream in streams:
                ptr = stream["pointer"]
                tokens = stream["tokens"]

                if ptr < len(tokens) and tokens[ptr][0] == "segment":
                    _append_segment_to_fused(
                        fused=fused,
                        source_circuit=stream["circ"],
                        instructions=tokens[ptr][1],
                        cindex_map=stream["cindex_map"],
                    )
                    stream["pointer"] += 1
                    progressed = True

            # Merge all next barriers into a single fused barrier
            merged_barrier_qubits = set()

            for stream in streams:
                ptr = stream["pointer"]
                tokens = stream["tokens"]

                if ptr < len(tokens) and tokens[ptr][0] == "barrier":
                    merged_barrier_qubits.update(tokens[ptr][1])
                    stream["pointer"] += 1
                    progressed = True

            if merged_barrier_qubits:
                fused.barrier(*[fused.qubits[q] for q in sorted(merged_barrier_qubits)])

            if not progressed:
                raise RuntimeError("Fusion made no progress, this indicates an internal logic error.")

        fused_circuits.append(fused)

    return fused_circuits, clbit_layout
