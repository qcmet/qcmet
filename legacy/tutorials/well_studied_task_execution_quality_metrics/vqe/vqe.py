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


import numpy as np
import openfermion as of
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import jordan_wigner
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.circuit.library import SlaterDeterminant
from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian



class FermiHubbardVQE:
    def __init__(self, nsites=2, U=0, t=1, shift_number=False):
        self.nsites = nsites
        self.U = U  # coulomb
        self.t = t  # tunneling
        self.shift_number = shift_number
        self.num_qubits = 2 * self.nsites
        self.params = ParameterVector("params", 0)


    def interaction_hamiltonian(self):
        i_hamiltonian = 0
        # interaction terms
        for i in range(self.nsites):
            i_hamiltonian += of.FermionOperator(
                "%d^ %d %d^ %d" % (i, i, i + self.nsites, i + self.nsites), self.U
            )
            if self.shift_number:
                i_hamiltonian -= of.FermionOperator("%d^ %d " % (i, i), self.U / 2)
                i_hamiltonian -= of.FermionOperator(
                    "%d^ %d " % (i + self.nsites, i + self.nsites), self.U / 2
                )
                i_hamiltonian += self.U / 4.0
        return i_hamiltonian


    def tight_binding_hamiltonian(self):
        tb_hamiltonian = 0
        for i in range(2 * self.nsites - 1):
            if i + 1 == self.nsites:
                continue
            else:
                tb_hamiltonian += of.FermionOperator(
                    "%d^ %d" % (i, i + 1), -self.t
                ) + of.FermionOperator("%d^ %d" % (i + 1, i), -self.t)

        return tb_hamiltonian


    def hamiltonian(self):
        return self.tight_binding_hamiltonian() + self.interaction_hamiltonian()


    def get_qc_param(self):
        size = self.params._size
        self.params.resize(size + 1)
        return self.params[-1]


    def prepare_initial_state_qiskit(self):
        m = np.zeros((2 * self.nsites, 2 * self.nsites))
        for i in range(self.nsites - 1):
            m[i, i + 1] = -self.t
            m[i + 1, i] = -self.t
            si = i + self.nsites
            m[si, si + 1] = -self.t
            m[si + 1, si] = -self.t

        quadratic_ham = QuadraticHamiltonian(m)
        (transformation_matrix, orbital_energies, transformed_constant) = (
            quadratic_ham.diagonalizing_bogoliubov_transform()
        )

        occupied_orbitals = [
            i for i in range(len(orbital_energies)) if orbital_energies[i] <= 0.0
        ]

        self.qc = SlaterDeterminant(transformation_matrix[list(occupied_orbitals)])
        self.qc.barrier()
        return self.qc


    def get_energy(self, circ):
        state = Statevector.from_int(0, 2 ** (2 * self.nsites))
        state = state.evolve(circ)

        sparse_ham = get_sparse_operator(jordan_wigner(self.hamiltonian()))
        psi = np.array(state)

        return np.real(np.conj(psi).T @ sparse_ham @ psi)


    def apply_vha(self):
        # uu layer
        p = self.get_qc_param()
        for i in range(self.nsites):
            self.qc.rzz(
                p,
                i,
                i + self.nsites,
            )
        self.qc.barrier()
        # u_h^1
        p = self.get_qc_param()
        for i in range(0, self.nsites - 1, 2):
            if self.nsites > 1:
                self.qc.append(XXPlusYYGate(p), [i, i + 1])
                self.qc.append(XXPlusYYGate(p), [i + self.nsites, i + 1 + self.nsites])
        self.qc.barrier()
        # u_h^2
        for i in range(1, self.nsites - 1, 2):
            if i:
                p = self.get_qc_param()
                if self.nsites > 1:
                    self.qc.append(XXPlusYYGate(p), [i, i + 1])
                    self.qc.append(
                        XXPlusYYGate(p), [i + self.nsites, i + 1 + self.nsites]
                    )


    def energy_expectation_operators(self):
        self.qc_list = {}
        for i in range(self.num_qubits - 1):
            if i + 1 == self.nsites:
                continue
            else:
                tqc = self.qc.copy()
                tqc.h(i)
                tqc.h(i + 1)
                tqc.measure_all()
                self.qc_list[f"x{i}"] = tqc

        for i in range(self.num_qubits - 1):
            if i + 1 == self.nsites:
                continue
            else:
                tqc = self.qc.copy()
                tqc.sdg(i)
                tqc.h(i)
                tqc.sdg(i + 1)
                tqc.h(i + 1)
                tqc.measure_all()
                self.qc_list[f"y{i}"] = tqc

        tqc = self.qc.copy()
        tqc.measure_all()
        self.qc_list["z"] = tqc


def get_energy(fhq, results, shots):
    t0 = 0
    for i in range(fhq.num_qubits - 1):
        if i + 1 == fhq.nsites:
            continue
        else:
            for basis in ["x", "y"]:
                key = f"{basis}{i}"
                for bitstring, counts in results[key].items():
                    parity = (int(bitstring[i]) + int(bitstring[i + 1])) % 2
                    t0_term = (-2 * parity + 1) * counts
                    t0 += t0_term
    t0 /= shots

    U = 0
    for i in range(fhq.nsites):
        for bitstring, counts in results["z"].items():
            parity_up = int(bitstring[i])
            parity_down = int(bitstring[i + fhq.nsites])
            parity = (parity_up + parity_down) % 2

            U_term = (
                (-2 * parity + 1) - (-2 * parity_up + 1) - (-2 * parity_down + 1)
            ) * counts
            U += U_term

    U /= shots
    t0 *= -fhq.t / 2
    U *= fhq.U / 4
    U += fhq.nsites * fhq.U / 4

    energy = t0 + U
    return energy
