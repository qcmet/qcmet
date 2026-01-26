"""Defines NoisySimulator device based on qiskit aer_simulator."""

import numpy as np
from qiskit.circuit.library import RXGate, RZGate, RZXGate, RZZGate
from qiskit_aer.noise import (
    NoiseModel,
    coherent_unitary_error,
    depolarizing_error,
    thermal_relaxation_error,
)

from qcmet.devices import AerSimulator


class NoisySimulator(AerSimulator):
    """Noisy AerSimulator using custom noise model."""

    def __init__(self, num_qubits = 5,
                overrotation_amount = np.pi / 100,
                detuning_amount = np.pi / 120,
                error_1q = 0.005,
                error_2q = 0.05,
                t1=50e3,
                t2=70e3,
                **kwargs
                 ):
        """Initialize a noisy Aer-based simulator with a configurable noise model.

        This constructor sets up a Qiskit Aer simulator that emulates realistic device
        imperfections. The noise model includes coherent over-rotation and detuning on
        single-qubit gates, depolarizing channels for single- and two-qubit gates, and
        optional thermal relaxation (T1/T2). The constructed NoiseModel and its basis
        gates are passed to the base AerSimulator.

        Args:
            num_qubits (int): Number of qubits used for assigning per-qubit thermal
                relaxation channels and for metadata. Default: 5.
            overrotation_amount (float): X-axis over-rotation applied after each single-qubit
                ``sx`` (π/2) gate, in radians. Default: π/100.
            detuning_amount (float): Z rotation applied after each single-qubit ``sx``
                gate to model detuning-induced phase error, in radians. Default: π/120.
            error_1q (float): Single-qubit depolarizing probability applied to ``sx`` gates.
                Must be in [0, 1]. Default: 0.005.
            error_2q (float): Two-qubit depolarizing probability applied to ``cx`` gates.
                Must be in [0, 1]. Default: 0.05.
            t1 (float | None): Energy relaxation time T1 in nanoseconds. If ``None`` or 0,
                thermal relaxation is disabled. Default: 50e3 (50 μs).
            t2 (float | None): Dephasing time T2 in nanoseconds. If provided larger than
                ``2*T1``, it is clipped to satisfy the physical constraint ``T2 ≤ 2*T1``.
                If ``None`` or 0, thermal relaxation is disabled. Default: 70e3 (70 μs).
            **kwargs: Additional keyword arguments passed through to the AerSimulator constructor.

        Behavior:
            - Builds a NoiseModel with:
                - coherent errors after ``sx`` gates (X over-rotation and Z detuning),
                - depolarizing noise on ``sx`` (1-qubit) and ``cx`` (2-qubit) gates,
                - optional T1/T2 thermal relaxation channels on ``id``, ``sx``, and ``cx``.
            - Clips ``t2`` to ``2*t1`` if larger, to maintain physical consistency.
            - Passes the noise model and its basis gates to the base ``AerSimulator``.
            - Stores ``num_qubits`` in ``self.properties['num_qubits']``.

        Attributes:
            num_qubits (int): Number of qubits.
            overrotation_amount (float): X over-rotation angle (rad) applied after ``sx``.
            detuning_amount (float): Z detuning angle (rad) applied after ``sx``.
            error_1q (float): Depolarizing probability for single-qubit gates.
            error_2q (float): Depolarizing probability for two-qubit gates.
            t1 (float | None): T1 coherence time (ns).
            t2 (float | None): T2 coherence time (ns), clipped to ``2*T1`` if needed.
            thermal_relaxation (bool): Whether thermal relaxation channels are enabled.
            properties (dict): Includes ``'num_qubits'`` entry set to ``num_qubits``.

        Example:
            Create a 7-qubit noisy simulator with custom depolarizing strengths:

                >>> sim = NoisySimulator(num_qubits=7, error_1q=5e-3, error_2q=5e-2)
                
        """
        self.num_qubits = num_qubits
        self.overrotation_amount = overrotation_amount
        self.detuning_amount = detuning_amount
        self.t1 = t1
        self.t2 = 2*self.t1 if t2>(2*self.t1) else t2
        self.error_1q = error_1q
        self.error_2q = error_2q
        self.t1 = t1
        self.t2 = t2
        
        super().__init__(self.noise_model(), self.noise_model().basis_gates, **kwargs)
        self.properties['num_qubits'] = num_qubits

    def noise_model(self):
        r"""Define custom noise model.

        T1 and T2 times are selected from a random normal distribution with a mean of
        50μs and 70μs respectively, and a standard deviation of 1μs for both. To ensure
        repeatability, a random seed is set such that the T1 times selected for each
        qubit remain the same.

        In order to apply amplitude and phase damping noise when executing a circuit,
        the gate times must also be known. The following gate times are used: time for
        idle gate (I) is 50ns, time for Rx(π/2) gate is 50ns, for CX gate is 300ns, and
        time for measurement is 1000ns. There is no noise applied on the Rz(theta) as
        it is modelled to be a virtual gate that is applied by adding a phase to the
        following gates.


        For all of the Rx(π/2) gates applied, after the ideal Rx(π/2) gate the
        following noise contributions are added:
        - an over-rotation around the x-axis of π/100 to simulate coherent
            calibration errors,
        - a rotation about the z-axis of π/120 to simulate the coherent phase error
            occurring due to the applied pulse being detuned from the qubit frequency
        - a depolarizing noise channel to approximate effectively averaged noise in
            a large quantum circuit. The depolarizing parameter used for this gate
            is gamma_D = 0.0005.

        For all of the 2-qubi CX gates applied, after the ideal CX gate the following
        noise contributions are added:
        - an exp^(-i*ZX\*theta_zx/2) operation and an exp^(-i\*ZZ*theta_zz/2) operation
            on the 2-qubit subspace the CX gate acts on.
        - The parameters theta_zx and theta_zz are both set to pi/100. The zx- and zz-
            rotation axes are chosen to reproduce some of the dominant sources of
            coherent error when applying a cross-resonance gate in superconducting qubits
        - a depolarizing noise channel, with depolarizing parameter gamma_D = 0.005.
            It is larger the value used for single qubit gates, as two-qubit gates
            typically have larger average errors.

        Returns:
            NoiseModel: custom noise_model

        """
        if self.t1 or self.t2 not in (None, 0):
            self.thermal_relaxation =  True
            thermal_relax_error_1q = thermal_relaxation_error(
                self.t1, self.t2, 10,
            )
        
        else:
            self.thermal_relaxation = False
        overrotation_amount = self.overrotation_amount
        detuning_amount = self.detuning_amount
        overrotation_unitary_1q = RXGate(overrotation_amount).to_matrix()
        detuning_unitary_1q = RZGate(detuning_amount).to_matrix()
        sx_gate_overrotation_error = coherent_unitary_error(overrotation_unitary_1q)
        sx_gate_detuning_error = coherent_unitary_error(detuning_unitary_1q)

        coherent_unitary_2q = RZXGate(overrotation_amount).to_matrix()
        zz_unitary_2q = RZZGate(overrotation_amount).to_matrix()
        coherent_unitary_2q_error = coherent_unitary_error(coherent_unitary_2q)
        zz_2q_error = coherent_unitary_error(zz_unitary_2q)

        noise_model = NoiseModel()
        error_1q = depolarizing_error(self.error_1q, 1)
        error_2q = depolarizing_error(self.error_2q, 2)
        noise_model.add_all_qubit_quantum_error(
            sx_gate_overrotation_error, ["sx"], warnings=False
        )
        noise_model.add_all_qubit_quantum_error(
            sx_gate_detuning_error, ["sx"], warnings=False
        )
        # noise_model.add_all_qubit_quantum_error(depolarizing_error(0.00005, 1), ["rz"])
        noise_model.add_all_qubit_quantum_error(error_1q, ["sx"], warnings=False)

        
        if self.thermal_relaxation:
            for j in range(self.num_qubits):
                noise_model.add_quantum_error(
                    thermal_relax_error_1q, "id", [j], warnings=False
                )
                noise_model.add_quantum_error(
                    thermal_relax_error_1q, "sx", [j], warnings=False
                )
                for k in range(self.num_qubits):
                    noise_model.add_quantum_error(
                        thermal_relax_error_1q.expand(thermal_relax_error_1q), "cx", [j, k]
                    )

        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"], warnings=False)
        noise_model.add_all_qubit_quantum_error(
            coherent_unitary_2q_error, ["cx"], warnings=False
        )
        noise_model.add_all_qubit_quantum_error(zz_2q_error, ["cx"], warnings=False)
        return noise_model
