# This code is part of QCMet.
# 
# (C) Copyright 2024 National Physical Laboratory and National Quantum Computing Centre 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Implementation of Cycle Benchmarking using qiskit
(https://doi.org/10.1038/s41467-019-13068-7)  
"""

import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pytket.qasm
import qiskit
from pytket.tailoring import PauliFrameRandomisation
from qiskit import Aer, transpile
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
from tqdm.autonotebook import tqdm
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error
)
import pathlib
from scipy.optimize import curve_fit
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
from _helpers.pygsti_experiment import *
from _helpers.circuit_submitter import *


class CycleBenchmarking:
    """
    Cycle Benchmarking class

    Attributes:
    ----------
    num_qubits: int
        Number of qubits to use
    num_rand_sequences_L: int
        Number of random sequences per pauli channel
    full_pauli_subspace: bool
        Whether to use the full pauli subspace - as number of qubits increases, the number of pauli channels increases exponentially so this is not practical for larger systems
    subspace_size_K: int
        Size of the subspace - only used if full_pauli_subspace is False
    num_shots: int
        Number of shots to use in simulations
    g_layer: qiskit.QuantumCircuit
        Repeated layered operation, called G in paper
    repetitions_list: list
        List of repetitions  [int, int]L this is the number of cycle repetitions to run.
    pauli_list: list
        List of pauli channels
    composite_process_fidelity: float
        Composite process fidelity - only set if get_composite_process_fidelity is called, and only valid if experiment is run.
    data: dict
        Dictionary of data of all experiments. This represents the results of the experiments.


    Methods:
    -------
    find_number_of_repeated_gates_for_identity - if you want to find the number of repeated gates to get to given identity, use this. It helps find the number of repetitions to use for the CB experiment.
    set_g_layer - sets the operation to benchmark
    generate_circuits - generates the circuits for the experiment and simulates results
    get_composite_process_fidelity - gets the composite process fidelity

    """

    pauli_gates = [XGate, YGate, ZGate, IGate]
    _pauli_labels = {XGate: "X", YGate: "Y", ZGate: "Z", IGate: "I"}
    _pauli_to_mat = {
        "X": XGate().to_matrix(),
        "Y": YGate().to_matrix(),
        "Z": ZGate().to_matrix(),
        "I": IGate().to_matrix(),
    }

    def __init__(
        self,
        num_qubits=2,
        num_rand_sequences_L=10,
        full_pauli_subspace=True,
        subspace_size_K=None,
        num_shots=1000,
        device_qubits=[0, 1],
        total_num_qubits=2,
        filepath=None
    ):
        """
        Initialises the cycle benchmarking class

        Parameters:
        ----------
        num_qubits: int
            Number of qubits to use
        num_rand_sequences_L: int
            Number of random sequences per pauli channel
        full_pauli_subspace: bool
            Whether to use the full pauli subspace - as number of qubits increases, the number of pauli channels increases exponentially so this is not practical for larger systems
        subspace_size_K: int
            Size of the subspace - only used if full_pauli_subspace is False
        num_shots: int
            Number of shots
        """
        self.device_qubits = device_qubits
        self.num_qubits = num_qubits
        self.total_num_qubits = total_num_qubits
        self.num_rand_sequences_L = num_rand_sequences_L
        if full_pauli_subspace:
            self.pauli_list = self._get_full_pauli_subspace()
            self.subspace_size_K = len(self.pauli_list)
        else:
            if subspace_size_K:
                self.subspace_size_K = subspace_size_K
                self.pauli_list = self._get_random_pauli_subspace()
        self.data = {}
        self.ideal_results = {}
        self.noisy_results = {}
        self.g_layer = None
        self.num_shots = num_shots
        if filepath:
            self.filepath = pathlib.Path(filepath)
        else:
            self.filepath = pathlib.Path(__file__).parent / "CB"
        self.filepath.mkdir(parents=True, exist_ok=True)

    def _get_full_pauli_subspace(self):
        p_list = list(itertools.product(self.pauli_gates, self.pauli_gates))
        p_list.remove(tuple(IGate for _ in range(self.num_qubits)))
        return p_list

    def _get_state_basis_changing_operations(self, qc, pauli, qubit):
        if pauli is XGate:
            qc.h(qubit)
        elif pauli is YGate:
            qc.h(qubit)
            qc.s(qubit)
        else:
            pass

    def _get_measurement_basis_changing_operations(self, qc, pauli, qubit):
        if pauli is XGate:
            qc.h(qubit)
        elif pauli is YGate:
            qc.sdg(qubit)
            qc.h(qubit)
        else:
            pass

    def _get_random_pauli_subspace(
        self,
    ):
        if self.subspace_size_K > (4**self.num_qubits - 1):
            raise ValueError("Subspace size larger than full subspace")
        p_list = np.random.choice(
            self.pauli_gates, size=(self.subspace_size_K, self.num_qubits)
        )
        p_list.remove(tuple(IGate for _ in range(self.num_qubits)))
        return p_list

    def get_random_n_qubit_pauli(self):
        return np.random.choice(self.pauli_gates, size=self.num_qubits)

    def set_g_layer(self, qiskit_circuit: qiskit.QuantumCircuit, repetitions_list):
        # if type(circuit_instruction) is qiskit.circuit.instruction.Instruction:
        self.g_layer = qiskit_circuit
        self.g_layer.barrier()
        self.repetitions_list = repetitions_list  # list of ints

    def _get_cycle_randomisations(self, repetitions):
        rep_cycle = self.g_layer.repeat(repetitions)
        rep_cycle = rep_cycle.decompose()
        qc_tket = pytket.qasm.circuit_from_qasm_str(rep_cycle.qasm())
        pfr = PauliFrameRandomisation()
        averaged_circuits = pfr.sample_circuits(qc_tket, self.num_rand_sequences_L)
        qcs = []
        for circ in averaged_circuits:
            qcs.append(
                qiskit.QuantumCircuit.from_qasm_str(
                    pytket.qasm.circuit_to_qasm_str(circ)
                )
            )
        return qcs

    def generate_circuits(self):
        for m in tqdm(self.repetitions_list):
            qc_m = {}
            for pauli_channel in tqdm(self.pauli_list, leave=False):
                qc_pc = []
                randomised_cycles = self._get_cycle_randomisations(repetitions=m)
                pc_label = "".join([self._pauli_labels[gate] for gate in pauli_channel])
                for j, circ in enumerate(randomised_cycles):
                    qc = qiskit.QuantumCircuit(self.total_num_qubits, self.total_num_qubits)

                    # basis changing operations
                    for i, pauli in enumerate(pauli_channel):
                        self._get_state_basis_changing_operations(qc, pauli, self.device_qubits[i])
                        qc.barrier(self.device_qubits[i])

                    # cycle operations including interleaved paulis
                    circ = circ.to_instruction()
                    circ_name = circ.name
                    qc.append(circ, [self.device_qubits[i] for i in range(circ.num_qubits)])
                    qc = qc.decompose(gates_to_decompose=circ_name)

                    # basis changing inverse
                    for i, pauli in enumerate(pauli_channel):
                        self._get_measurement_basis_changing_operations(qc, pauli, self.device_qubits[i])
                    qc.measure(self.device_qubits[0], self.device_qubits[0])
                    qc.measure(self.device_qubits[1], self.device_qubits[1])
                    # print(qc)
                    qc_pc.append(
                        {
                            "qc": qc.qasm(),
                            "counts": {
                                "noisy": 0,
                            },
                            "_expectation": {
                                "noisy": 0,
                            }
                        }
                    )

                qc_m[pc_label] = qc_pc
            self.data[m] = qc_m

        self.save_data(self.data, "data_before_run")
    
    def run_circuits(self, submitter):
        self.submitter = submitter

        circuits = []
        for m in self.repetitions_list:
            for pauli_channel in self.pauli_list:
                pc_label = "".join([self._pauli_labels[gate] for gate in pauli_channel])
                for each_qc_pc in self.data[m][pc_label]:
                    circuits.append(each_qc_pc["qc"])
        
        tasks = self.submitter.submit_circuits(shots=self.num_shots, qasm_strs=circuits)
        all_counts = self.submitter.retrieve_counts(wait=True)
        i = 0
        for m in self.repetitions_list:
            for pauli_channel in self.pauli_list:
                pc_label = "".join([self._pauli_labels[gate] for gate in pauli_channel])
                for each_qc_pc in self.data[m][pc_label]:
                    counts_noisy = all_counts[i]
                    each_qc_pc["counts"]["noisy"] = counts_noisy
                    exp_noisy = self._expectation(pc_label, counts_noisy)
                    each_qc_pc["_expectation"]["noisy"] = exp_noisy
                    i += 1
        self.save_data(self.data, "data")

    def save_data(self, data, name):
        with open(self.filepath / f"{name}.json", "w") as fp:
            json.dump(data, fp, indent=True)

    def _expectation(self, pauli_string, counts):
        exp = 0
        for state, counts in counts.items():
            parity = 1
            for bit, pauli in zip(state, pauli_string):
                if pauli != "I":
                    if int(bit) == 1:
                        parity *= -1
            exp += parity * counts
        return exp / self.num_shots

    def _pauli_string_to_matrix(self, pauli_string):
        pauli = self._pauli_to_mat[pauli_string[0]].astype(np.complex128)
        for pauli_str in pauli_string[1:]:
            pauli = np.kron(pauli, self._pauli_to_mat[pauli_str].astype(np.complex128))
        return pauli / len(pauli_string)

    def get_ptm_elements(self, data):
        ptm_elements = {}
        for pauli in data.keys():
            ptm_element = 0
            for i in data[pauli]:
                ptm_element += i["_expectation"]["noisy"]
            ptm_element /= self.num_rand_sequences_L
            ptm_elements[pauli] = ptm_element
        return ptm_elements

    def get_all_cycle_ptm_elements(self):
        ptm_elements = {}
        for m, datadict in self.data.items():
            ptm_elements[m] = self.get_ptm_elements(datadict)

        self.ptm_elements = ptm_elements
        return self.ptm_elements

    def get_cycle_fidelities(self):
        fids = []
        for i in self.repetitions_list:
            fids.append(np.sum(np.asarray(list(self.ptm_elements[i].values()))))
        fids = [(i + 1) / 16 for i in fids]
        self.cycle_fidelities = fids
        print(fids)
        return self.cycle_fidelities

    def fit_func(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def get_fidelity_fit(self):
        # Do this to add soft constraint that fidelity=1 when m=0
        repetitions_list =  np.append(self.repetitions_list, np.zeros_like(self.repetitions_list))
        cycle_fidelities = np.append(self.cycle_fidelities, np.ones_like(self.cycle_fidelities))
        self.popt, self.pcov = curve_fit(
            self.fit_func, repetitions_list, cycle_fidelities
        )
        return self.popt, self.pcov
    
    def plot_pauli_eigv_vs_cycle_length(self):
        plt.figure()
        m = self.repetitions_list[0]
        for pauli in self.ptm_elements[m].keys():
            plt.scatter(
                self.repetitions_list,
                [self.ptm_elements[i][pauli] for i in self.repetitions_list],
                label=pauli,
            )
            plt.plot(
                self.repetitions_list,
                [self.ptm_elements[i][pauli] for i in self.repetitions_list],
            )
        plt.legend()
        plt.ylabel("PTM diagonal element value")
        plt.xlabel("Number of cycles")
        plt.savefig(self.filepath / "ptm_vs_cycle_length.png")
        plt.close()

    def plot_fidelity_vs_cycle_length(self):
        plt.figure()

        plt.scatter(self.repetitions_list, self.cycle_fidelities)
        fake_xdata = np.linspace(
            self.repetitions_list[0], self.repetitions_list[-1], 100
        )
        plt.plot(
            fake_xdata,
            self.fit_func(fake_xdata, *self.popt),
            "r-",
            label="fit: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(self.popt),
        )
        # plt.show()

        plt.ylabel("Fidelity")
        plt.xlabel("Number of cycles")
        plt.savefig(self.filepath / "fidelity_vs_cycle_length.png")
        plt.close()

    def fit_and_save_fit_and_figs(self):
        self.get_all_cycle_ptm_elements()
        self.get_cycle_fidelities()
        self.get_fidelity_fit()
        self.get_composite_process_fidelity_from_fit()
        self.plot_pauli_eigv_vs_cycle_length()
        self.plot_fidelity_vs_cycle_length()
        self.save_data(self.ptm_elements, "ptm_elements")
        self.save_data(self.cycle_fidelities, "cycle_fidelities")
        self.save_data(
            {"popt": self.popt.tolist(), "pcov": self.pcov.tolist()}, "popt_pcov"
        )
        self.save_data(self.composite_process_fidelity, "composite_process_fidelity")
    
    def get_composite_process_fidelity_from_fit(self):
        self.composite_process_fidelity = self.fit_func(1, *self.popt)
        return self.composite_process_fidelity

    def get_composite_process_fidelity(self):
        ptms = [{} for _ in self.repetitions_list]
        num_depths = len(self.repetitions_list)
        sum_pairs = [1 for _ in range(num_depths * (num_depths - 1) // 2)]

        for pauli in self.data[self.repetitions_list[0]].keys():
            temp_sums = [0 for _ in self.repetitions_list]
            for i, m in enumerate(self.repetitions_list):
                for res in self.data[m][pauli]:
                    temp_sums[i] += res['_expectation']['noisy']
                temp_sums[i] /= self.num_rand_sequences_L
                ptms[i][pauli] = temp_sums[i]
            k = 0
            for i in range(num_depths):
                for j in range(i+1, num_depths):
                    sum_pairs[k] += (temp_sums[j] / temp_sums[i]) ** (1 / (self.repetitions_list[j] - self.repetitions_list[i]))
                    k += 1

        for i in range(num_depths):
            print(self.repetitions_list[i], ptms[i])

        k = 0
        self.composite_process_fidelity = {}
        for i in range(num_depths):
            for j in range(i+1, num_depths):
                composite_process_fidelity = sum_pairs[k] / (self.subspace_size_K + 1)
                pair_str = f"depth {self.repetitions_list[i]} and {self.repetitions_list[j]}"
                print(f"Composite process fidelity between {pair_str} is {composite_process_fidelity:.4f}")
                self.composite_process_fidelity[pair_str] = composite_process_fidelity
                k += 1
        
        self.save_data(self.composite_process_fidelity, f"composite_process_fidelity")
        return self.composite_process_fidelity


if __name__ == "__main__":
    num_qubits = 2
    num_shots = 100
    num_rand_sequences_L = 60
    total_num_qubits = 4
    method = "ratio" # choose between "fit" and "ratio"
    device_name = "noisy_sim"

    submitter = CircuitSubmitter(benchmark_name="cb", device_name=device_name)
    # Uncomment the following lines if you are using a noisy simulator and would like to change the noise model
    # noise_model = None
    # submitter.device.noise_model = noise_model

    CB = CycleBenchmarking(
        num_qubits=num_qubits,
        num_shots=num_shots,
        num_rand_sequences_L=num_rand_sequences_L,
        device_qubits=[0, 1],
        total_num_qubits=total_num_qubits,
        filepath=submitter.benchmark_path,
    )

    interleaved_layer = qiskit.QuantumCircuit(num_qubits, name="interleaved_layer")
    interleaved_layer.cx(1, 0)
    CB.set_g_layer(interleaved_layer, [2, 4, 8, 10])

    CB.generate_circuits()
    CB.run_circuits(submitter=submitter)
    if method == "fit":
        CB.fit_and_save_fit_and_figs()
    elif method == "ratio":
        CB.get_composite_process_fidelity()
