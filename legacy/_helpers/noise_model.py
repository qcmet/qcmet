from qiskit.circuit.library import RZXGate, RZGate, RXGate, RZZGate
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
    coherent_unitary_error
)
import numpy as np

def custom_noise_model(num_qubits = 4, seed = 0):

    np.random.seed(seed)

    T1s = np.random.normal(50e3, 1e3, num_qubits) # Sampled from normal distribution mean 50 microsec
    T2s = np.random.normal(70e3, 1e3, num_qubits)  # Sampled from normal distribution mean 50 microsec

    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])


    # Instruction times (in nanoseconds)
    time_rz = 0   # virtual gate
    time_sx = 50  # (single X90 pulse)
    time_x = 100 # (two X90 pulses)
    time_cx = 300
    time_reset = 1000  # 1 microsecond
    time_measure = 1000 # 1 microsecond


    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_rz)
                for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_sx)
                for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_x)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]

    noise_model = NoiseModel()

    overrotation_amount = np.pi/100
    detuning_amount = np.pi/120
    overrotation_unitary_1q = RXGate(overrotation_amount).to_matrix()
    detuning_unitary_1q = RZGate(detuning_amount).to_matrix()
    sx_gate_overrotation_error = coherent_unitary_error(overrotation_unitary_1q)
    sx_gate_detuning_error = coherent_unitary_error(detuning_unitary_1q)

    coherent_unitary_2q = RZXGate(overrotation_amount).to_matrix()
    zz_unitary_2q = RZZGate(overrotation_amount).to_matrix()
    coherent_unitary_2q_error = coherent_unitary_error(coherent_unitary_2q)
    zz_2q_error = coherent_unitary_error(zz_unitary_2q)


    # Add errors to noise model
    noise_model = NoiseModel()
    for j in range(num_qubits):
        noise_model.add_quantum_error(errors_reset[j], "reset", [j]) # maybe specify amplitude damping and dephasing separately
        noise_model.add_quantum_error(errors_measure[j], "measure", [j])
        noise_model.add_quantum_error(errors_u1[j], "rz", [j])
        noise_model.add_quantum_error(errors_u2[j], "sx", [j])
        # noise_model.add_quantum_error(errors_u3[j], "x", [j])
        noise_model.add_quantum_error(sx_gate_overrotation_error, ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(sx_gate_detuning_error, ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(depolarizing_error(0.0005,1), ['sx'], [j], warnings=False)
        noise_model.add_quantum_error(errors_u2[j], "id", [j])
        #add detuning error maybe tlak to abhishek
        for k in range(num_qubits):
            noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])
            noise_model.add_quantum_error(depolarizing_error(0.005,2), ["cx"], [j, k], warnings=False)
            noise_model.add_quantum_error(coherent_unitary_2q_error, ["cx"], [j, k], warnings=False)
            noise_model.add_quantum_error(zz_2q_error, ["cx"], [j, k], warnings=False)

    return noise_model

if __name__=="__main__":
    nm = custom_noise_model()
    print(nm.basis_gates)