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

from qiskit.quantum_info import random_clifford
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit
from qiskit.synthesis import synth_clifford_full
import matplotlib.pyplot as plt
import numpy as np


class RandomizedBenchmarking:

    def __init__(
        self,
        n_qubits=2,
        n_cliffords=[0, 1, 4, 8, 16, 32, 64, 128, 256],
        samples_per_depth=10,
        interleaved_circuit=None,
    ):
        self.n_qubits = n_qubits
        self.n_cliffords = n_cliffords
        self.samples_per_depth = samples_per_depth

        self.interleaved_circuit = interleaved_circuit

    def random_clifford_circuit(self, depth):
        base_circuit = QuantumCircuit(self.n_qubits)
        for _ in range(depth):
            random_clifford_operator = random_clifford(self.n_qubits)
            clifford_circuit = synth_clifford_full(
                random_clifford_operator, method="AG"
            )
            base_circuit.compose(clifford_circuit, inplace=True)
            base_circuit.barrier()
            if self.interleaved_circuit is not None:
                base_circuit.compose(self.interleaved_circuit, inplace=True)
                base_circuit.barrier()
        base_circuit.compose(base_circuit.inverse(), inplace=True)
        base_circuit.measure_all()
        return base_circuit

    def generate_all_circuits(self):
        self.circuits = {}
        for sequence_length in self.n_cliffords:
            self.circuits[sequence_length] = []
            for _ in range(self.samples_per_depth):
                self.circuits[sequence_length].append(
                    self.random_clifford_circuit(sequence_length)
                )
        return self.circuits

    def load_counts(self, all_counts, shots):
        if type(all_counts) != dict:
            raise ValueError(
                "Counts must be a dictionary of lists, with the keys being sequence lengths"
            )
        self.all_counts = all_counts
        probs_0 = []
        key_for_ground_state = "".join("0" for _ in range(self.n_qubits))
        for _, counts_list in all_counts.items():
            counts_0 = []
            for count in counts_list:
                counts_0.append(count[key_for_ground_state] / shots)
            probs_0.append(counts_0)
        self.probs_0 = np.asarray(probs_0)

    @staticmethod
    def rb_fit_func(m, alpha, a_0, b_0):
        return a_0 * np.power(alpha, m) + b_0

    def get_fit(self, initial_guess=(0.99, 0.8, 0)):
        array_of_depths = np.array(
            [
                [float(i) for _ in range(self.samples_per_depth)]
                for i in self.n_cliffords
            ]
        )
        self.fit, self.pcov = curve_fit(
            self.rb_fit_func,
            array_of_depths.ravel(),
            self.probs_0.ravel(),
            p0=initial_guess,
            bounds=((0.5, 0, 0), (1, 1, 1 / (2**self.n_qubits))),
        )
        return self.fit

    def get_average_gate_error(self):
        alpha = self.fit[0]
        d = 2**self.n_qubits
        self.average_gate_error = 1 - alpha - (1 - alpha) / d
        print(
            f"The Clifford randomized benchmarking average gate error is {self.average_gate_error}"
        )
        return self.average_gate_error

    def plot_rb_results(self, label=None, color = '#1f77b4'):
        violin_parts = plt.violinplot(
            self.probs_0.T,
            positions=self.n_cliffords,
            widths=[1.5 * np.log2(i) if i > 1 else 1 + i for i in self.n_cliffords],
            showextrema=False,
        )
        if label is None:
            label = "RB fit"
        xxs = np.linspace(0, self.n_cliffords[-1],1000)
        l_plot, = plt.plot(
            xxs,
            self.rb_fit_func(xxs, *self.fit), ls='--',
            label=label, color = color
        )
        plt.scatter(self.n_cliffords, np.mean(self.probs_0, axis=1), marker='x', s=30, color = l_plot.get_color())
        plt.xlim((0, self.n_cliffords[-1]+10))
        y_min = (np.min(self.probs_0)) - 0.02 if ((np.min(self.probs_0)) - 0.02) >=0 else 0 
        plt.ylim((y_min, 1))
        plt.xlabel(r'$m$')
        plt.ylabel(r"$p_{\mathrm{survival}}$")
        for vp in violin_parts['bodies']:
            vp.set_facecolor(l_plot.get_color())
            vp.set_alpha(0.3)

class InterleavedRandomizedBenchmarking:
    def __init__(
        self,
        n_qubits=2,
        n_cliffords=[
            0,
            2,
            4,
            8,
            16,
            32,
            64,
        ],
        samples_per_depth=10,
        interleaved_circuit=None,
    ):
        self.n_qubits = n_qubits
        self.n_cliffords = n_cliffords
        self.samples_per_depth = samples_per_depth
        if interleaved_circuit is None:
            raise ValueError("Must define interleaved gate as qiskit quantum circuit")
        self.interleaved_circuit = interleaved_circuit
        self.rb_base = RandomizedBenchmarking(
            self.n_qubits, self.n_cliffords, self.samples_per_depth
        )
        self.rb_interleaved = RandomizedBenchmarking(
            self.n_qubits,
            self.n_cliffords,
            self.samples_per_depth,
            self.interleaved_circuit,
        )

    def generate_all_circuits(self):
        self.circs_base, self.circs_interleaved = (
            self.rb_base.generate_all_circuits(),
            self.rb_interleaved.generate_all_circuits(),
        )
        return self.circs_base, self.circs_interleaved

    def load_counts(self, base_counts, interleaved_counts, shots):
        self.rb_base.load_counts(base_counts, shots=shots)
        self.rb_interleaved.load_counts(interleaved_counts, shots=shots)

    def get_fit(self):
        self.fit_base = self.rb_base.get_fit()
        self.fit_interleaved = self.rb_interleaved.get_fit()
        return self.fit_base, self.fit_interleaved

    def get_interleaved_gate_error(self):
        alpha_g = self.fit_interleaved[0]
        alpha = self.fit_base[0]
        self.interleaved_gate_error = (
            (2**self.n_qubits - 1) * (1 - (alpha_g / alpha)) / (2**self.n_qubits)
        )
        print(
            f"The interleaved Clifford randomized benchmarking average gate error is {self.interleaved_gate_error}"
        )
        return self.interleaved_gate_error

    def get_base_rb_average_gate_error(self):
        self.rb_base.get_average_gate_error()

    def plot_rb_results(self):
        self.rb_base.plot_rb_results(label="Clifford RB")
        
        self.rb_interleaved.plot_rb_results(label="Interleaved Clifford RB", color='#ff7f0e')
