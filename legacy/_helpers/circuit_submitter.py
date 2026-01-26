from cirq.contrib.qasm_import import circuit_from_qasm
from cirq_ionq.ionq_native_gates import GPIGate, GPI2Gate, MSGate
import cirq_ionq
import numpy as np
import cirq
import qiskit
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
from _helpers.backend_helpers import *
from braket.aws import AwsQuantumTask
from braket.tasks.local_quantum_task import LocalQuantumTask
from braket.ir.openqasm import Program
import os
import sys
from pathlib import Path
from datetime import datetime
import time
import json


class CircuitSubmitter():
    """A central interface for individual benchmarks to submit circuits."""
    def __init__(self, benchmark_name: str, device_name: str = "noisy_sim"):
        """
        Args:
            benchmark_name: the name of the benchmark.
            device_name: the name of the device.
                Choose among "simulator", "noisy_sim", "noiseless_sim", "Aria".
        """
        self.benchmark_name = benchmark_name
        self.device_name = device_name
        self.initialise()
        self.circuits_padded = 0
        self.tasks = []

    def initialise(self):
        """Initialise the folder structure and record device calibration details."""
        # Create folders for the device and date
        self.backend = get_backend_helper(self.device_name)
        self.device_date_path = os.path.dirname(os.path.realpath(__file__)) + "/../hardware_runs/" + \
                            f"{self.device_name}/{datetime.today().strftime('%Y-%m-%d')}"
        Path(self.device_date_path).mkdir(parents=True, exist_ok=True)
        
        # Record device calibration details
        json_path = self.device_date_path + "/device_calibration_details.json"
        with open(json_path, "w+") as f:
            json.dump(self.backend.get_device_calibration(), f)
        
        # Create folders for the benchmark
        self.benchmark_path = self.device_date_path + '/' + self.benchmark_name
        Path(self.benchmark_path).mkdir(parents=True, exist_ok=True)

        self.circuits_path = self.benchmark_path + '/circuits'
        Path(self.circuits_path).mkdir(parents=True, exist_ok=True)
    
    def transpile(self, circuits: list[QuantumCircuit], verbatim: bool = True) -> list[Circuit]:
        """Transpile the circuits to use native gates of self.device_name."""
        if self.device_name != 'Aria' and self.device_name !='Harmony':
            braket_circuits = transpile(circuits, backend=self.backend.get_qiskit_backend(), basis_gates=self.backend.get_basis_gates(),
                                optimization_level=0)
            braket_circuits = list(convert_qiskit_to_braket_circuits(braket_circuits))
        else:
            braket_circuits = []
            for circuit in circuits:
                ion_circuit = self.traspile_to_ionq_native_gates(circuit)
                braket_circ = Circuit()
                for op in ion_circuit.all_operations():
                    if isinstance(op.gate, cirq_ionq.ionq_native_gates.GPI2Gate): 
                        braket_circ.gpi2(op.qubits[0].x, op.gate.phi * 2 * np.pi)
                    if isinstance(op.gate, cirq_ionq.ionq_native_gates.GPIGate):
                        braket_circ.gpi(op.qubits[0].x, op.gate.phi * 2 * np.pi)
                    if isinstance(op.gate, cirq_ionq.ionq_native_gates.MSGate):
                        braket_circ.ms(op.qubits[0].x, op.qubits[1].x, op.gate.phi0 * 2 * np.pi, op.gate.phi1 * 2 * np.pi)
                braket_circuits.append(braket_circ)

        # Add two X gates or GPi gates in case the circuit is empty, to avoid AWS submission errors
        for c in braket_circuits:
            # For GST on IonQ: add GPi gates on empty q0
            if "gst" in self.benchmark_name.lower() and self.device_name in ["Aria", "Harmony"]:
                q0_is_not_empty = False
                for ins in c.instructions:
                    for qubit in ins.target:
                        if qubit == 0:
                            q0_is_not_empty = True
                            break
                    if q0_is_not_empty:
                        break
                if not q0_is_not_empty:
                    c.gpi(0, 0)
                    c.gpi(0, 0)
                    self.circuits_padded += 1

            if c.depth == 0:
                if self.device_name not in ["Harmony", "Aria"]:
                    c.x(0)
                    c.x(0)
                else:
                    c.gpi(0, 0)
                    c.gpi(0, 0)
                self.circuits_padded += 1

        if verbatim:
            braket_circuits = wrap_circuits_in_verbatim_box(braket_circuits)
        print(f"There are {self.circuits_padded}/{len(braket_circuits)} circuits that got added two x or gpi gates to avoid errors") 
        
        self.circuits_padded = 0
        return braket_circuits

    def estimate_cost_and_ask(self, circuits_len: int, shots: int, skip_asking: bool = False, print_summary: bool = True):
        """Estimate the cost of the circuits, and ask if you want to proceed."""
        cost_per_circuit, cost_per_shot = self.backend.get_costs()
        
        cost = circuits_len * (cost_per_circuit + shots * cost_per_shot)
        if print_summary:
            print(f"Ready to run {circuits_len} circuits on {self.device_name} with {shots} shots.\n"
              f"Cost: {circuits_len} * (${cost_per_circuit} + {shots} * ${cost_per_shot}) = ${cost:.2f}.")
        sys.stdout.flush()
        if not skip_asking:
            response = input("Proceed? (y): ")
            if response.lower() not in ["yes", 'y']:
                raise Exception("Terminated")
            else:
                print('Submitting circuits...')
    
    def submit_circuits(self, shots: int, verbatim: bool = True, skip_asking: bool = False, skip_transpilation: bool = False, print_summary: bool = True,
                        braket_circuits: list[Circuit] = None, qasm_strs: list[str] = None, 
                        qasm_paths: list[str] = None,  inputs: dict[str, float] = None) -> Union[list[AwsQuantumTask], list[LocalQuantumTask]]:
        """Submit circuits to run on hardware.

        Args: 
            shots: number of shots for each circuit.
            verbatim: whether to force verbatim circuits with no optimisation.
            skip_asking: whether to skip asking during circuits cost estimation.
            skip_transpilation: whether to skip transpilation of circuits, to directly run the braket_circuits.
            braket_circuits: a list of braket.Circuit. Must provide this if skip_transpilation is True.
            qasm_strs: a list of qasm strings, each representing one circuit.
            qasm_paths: a list of qasm file paths, each representing one circuit.
        One and only one of qasm_strs or qasm_paths should be used.

        Returns:
            tasks: a list of tasks that are submitted.
        """
        if not skip_transpilation:
            # Get qiskit circuits from qasm
            if qasm_strs is not None:
                circuits = [QuantumCircuit.from_qasm_str(string) for string in qasm_strs]
            elif qasm_paths is not None:
                circuits = [QuantumCircuit.from_qasm_file(path) for path in qasm_paths]
            else:
                raise ValueError("One and only one of qasm_strs or qasm_paths should be used")

            if self.device_name == "OQCDirect":
                circuits = [c.qasm() for c in circuits]
            
            elif self.device_name in ("noisy_sim", "noisy_sim_with_shots"):
                # Don't convert to braket circuits for running on qiskit AerSimulator
                circuits = qiskit.transpile([c for c in circuits], basis_gates = self.backend.device.noise_model.basis_gates)
            
            elif self.device_name == "noiseless_sim":
                
                # Don't convert to braket circuits for running on qiskit AerSimulator
                pass
            else:
                # Transpile circuits if not using OQC Direct access
                circuits = self.transpile(circuits, verbatim)
        else:
            if self.backend.__class__.__bases__[0].__name__ == 'AwsBackendHelper': 
                if braket_circuits is None:
                    if qasm_strs is None:
                        raise ValueError("To skip transpilation you must provide qasm_strs")
                circuits = braket_circuits
            if self.device_name == "noisy_sim" or "noisy_sim_with_shots":
                if qasm_strs is not None:
                    circuits = [QuantumCircuit.from_qasm_str(string) for string in qasm_strs]


        # Prompts user to confirm submitting circuits
        self.estimate_cost_and_ask(len(circuits), shots, skip_asking, print_summary=print_summary)

        # Submit circuits
        batch_size = 100
        tasks = []
        for circuits_this_batch in [circuits[i:i + batch_size] for i in range(0, len(circuits), batch_size)]:
            if self.device_name == "simulator":
                tasks_this_batch = [self.backend.device.run(circuit, shots=shots) for circuit in circuits_this_batch]
            else:
                task_batch = self.backend.device.run_batch(circuits_this_batch, shots=shots, max_parallel=batch_size, inputs=inputs)
                tasks_this_batch = task_batch.tasks
            for task in tasks_this_batch:
                cid = task.id.replace('/', '=').replace(':', '_') # Some magic symbols to make a good folder name
                task_path = self.circuits_path + '/' + cid
                Path(task_path).mkdir(parents=True, exist_ok=True)
                tasks.append(task)

                # For braket simulator: directly dump the results
                if self.device_name == "simulator":
                    with open(self.circuits_path + f"/{task.id}/results.json", "w+") as f:
                        json.dump(task.result(), f, indent=4, default=lambda o: str(o.tolist()) if isinstance(o, np.ndarray) else o.__dict__)
        if print_summary:
            print("Circuits have been submitted")
        self.tasks = tasks
        return tasks
    
    def retrieve_counts(self, circuit_ids: list[str] = None, wait: bool = True, print_timestamp_when_done = True):
        """Try to get the results of circuits in terms of counts.

        Args:
            circuit_ids: the list of circuit ids to retrieve results for. 
                If not provided, will retrieve the most recent run.
            wait: if set to False, will throw an exception if not all circuits are completed.
                If set to True, will attempt to wait (i.e. sleep) until completion.
        
        Returns:
            all_counts: a list with each element being a dict of bitstring counts for each circuit.
        """
        if not circuit_ids:
            if len(self.tasks) != 0:
                tasks = self.tasks
            else:
                # Sort the folders according to creation time
                circuit_ids = sorted(os.listdir(self.circuits_path), key=lambda f: os.path.getctime(os.path.join(self.circuits_path, f)))

        if self.device_name == "simulator":
            counts = [json.load(open(self.circuits_path + f"/{cid}/results.json"))["measurement_counts"] for cid in circuit_ids]
            return counts
        elif self.device_name == "noisy_sim" or "noisy_sim_with_shots":
            tasks = self.tasks
        elif self.device_name == "noiseless_sim":
            tasks = self.tasks
        elif self.device_name == "OQCDirect":
            tasks = [self.backend.device.client.get_task(cid) for cid in circuit_ids]
        else:
            tasks = [AwsQuantumTask(arn=cid.replace('=', '/').replace('_', ':')) for cid in circuit_ids]

        # Check all tasks are completed
        states = [task.state() for task in tasks]
        while not all(state == "COMPLETED" for state in states):
            if not wait:
                raise Exception(f"Only {states.count('COMPLETED')}/{len(states)} circuits are finished")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {states.count('COMPLETED')}/{len(states)} circuits are finished")
            time.sleep(10)
            # Refresh the states
            for i in range(len(states)):
                if states[i] != "COMPLETED":
                    states[i] = tasks[i].state()
        if print_timestamp_when_done:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} All circuits are finished")

        # Get all_counts
        for task in tasks:
            with open(self.circuits_path + f"/{task.id.replace('/', '=').replace(':', '_')}/results.json", "w+") as f:
                json.dump(task.result(), f, indent=4, default=lambda o: str(o.tolist()) if isinstance(o, np.ndarray) else o.__dict__)
        all_counts = [task.result().measurement_counts for task in tasks]
        return all_counts
    
    def convert_counts_to_qiskit(self, counts: dict):
        """Convert measurement counts into qiskit's reversed order.
        E.g. braket gives {'001': 400, '110': 600} but qiskit wants {'100': 400, '011': 600}
        """
        counts = {k[::-1]: v for k, v in counts.items()}
        return counts


    def compile_to_native_json(self, circuit):
        qubit_phase = [0] * 32
        op_list = []
        for op in circuit.all_operations():
            if type(op.gate) == cirq.ops.common_gates.Rz:
                qubit_phase[op.qubits[0].x] = (qubit_phase[op.qubits[0].x]-op.gate._rads/(2*np.pi))%1

            elif type(op.gate)==cirq.ops.common_gates.Ry:
                if abs(op.gate._rads-0.5*np.pi)<1e-6:
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0.25)%1).on(op.qubits[0])
                    )
                elif abs(op.gate._rads+0.5*np.pi)<1e-6:
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0.75)%1).on(op.qubits[0])
                        )
                elif abs(op.gate._rads-np.pi)<1e-6:
                    op_list.append(
                        GPIGate(phi=(qubit_phase[op.qubits[0].x]+0.25)%1).on(op.qubits[0])
                    )
                elif abs(op.gate._rads+np.pi)<1e-6:
                    op_list.append(
                        GPIGate(phi=(qubit_phase[op.qubits[0].x]+0.75)%1).on(op.qubits[0])
                    )
                else:
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0)%1).on(op.qubits[0])
                    )
                    qubit_phase[op.qubits[0].x]=(qubit_phase[op.qubits[0].x]-op.gate._rads/(2*np.pi))%1
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0.5)%1).on(op.qubits[0])
                    )

            elif type(op.gate)==cirq.ops.common_gates.Rx:
                if abs(op.gate._rads-0.5*np.pi)<1e-6:
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0)%1).on(op.qubits[0])
                    )
                elif abs(op.gate._rads+0.5*np.pi)<1e-6:
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0.5)%1).on(op.qubits[0])
                    )
                elif abs(op.gate._rads-np.pi)<1e-6:
                        op_list.append(
                        GPIGate(phi=(qubit_phase[op.qubits[0].x]+0)%1).on(op.qubits[0])
                    )
                elif abs(op.gate._rads+np.pi)<1e-6:
                    op_list.append(
                        GPIGate(phi=(qubit_phase[op.qubits[0].x]+0.5)%1).on(op.qubits[0])
                    )
                else:
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0.75)%1).on(op.qubits[0])
                    )
                    qubit_phase[op.qubits[0].x]=(qubit_phase[op.qubits[0].x]-op.gate._rads/(2*np.pi))%1
                    op_list.append(
                        GPI2Gate(phi=(qubit_phase[op.qubits[0].x]+0.25)%1).on(op.qubits[0])
                    )
            
            elif type(op.gate)==cirq.ops.parity_gates.XXPowGate:
                if op.gate.exponent>0:
                    op_list.append(
                        MSGate(
                            phi0=qubit_phase[op.qubits[0].x], 
                            phi1=qubit_phase[op.qubits[1].x]
                        ).on(op.qubits[0],op.qubits[1])
                    )
                else:
                    op_list.append(
                        MSGate(
                            phi0=qubit_phase[op.qubits[0]], 
                            phi1=(qubit_phase[op.qubits[1].x]+0.5)%1
                        ).on(op.qubits[0],op.qubits[1])
                    )
        return op_list

    def map_func(self, op: cirq.Operation, _: int) -> cirq.OP_TREE:
        if op.gate == cirq.CNOT:
            yield cirq.ry(np.pi / 2).on(op.qubits[0])
            yield cirq.XXPowGate(exponent=0.5).on(op.qubits[0], op.qubits[1])
            yield cirq.rx(-np.pi / 2).on(op.qubits[0])
            yield cirq.rx(-np.pi / 2).on(op.qubits[1])
            yield cirq.ry(-np.pi / 2).on(op.qubits[0])
        else:
            yield op

    def transpile_qiskit_to_cirq_for_ionq(self, qiskit_circuit):
        circ = qiskit.transpile(qiskit_circuit, basis_gates=['rx', 'ry', 'rz', 'cx', 'id'])
        rb = RemoveBarriers()
        circ_out = rb(circ)
        circ_out_qasm = circ_out.qasm()
        circ_qc = circuit_from_qasm(circ_out_qasm)
        rxx_circ_qc = cirq.map_operations_and_unroll(circ_qc, map_func=self.map_func)
        num_qubits = len(list(circ_qc.all_qubits()))
        qubits = cirq.LineQubit.range(num_qubits)
        qubit_map = dict()
        for qubit in circ_qc.all_qubits():
            qubit_map[qubit] = qubits[int(qubit.name.split('_')[-1])]
        new_qc = rxx_circ_qc.transform_qubits(qubit_map)
        return new_qc
    
    def traspile_to_ionq_native_gates(self, qiskit_qc):
        cirq_qc = self.transpile_qiskit_to_cirq_for_ionq(qiskit_qc)
        new_cirq_qc = self.compile_to_native_json(cirq_qc)
        circuit = cirq.Circuit()
        for i in new_cirq_qc:
            circuit.append(i)
        return circuit