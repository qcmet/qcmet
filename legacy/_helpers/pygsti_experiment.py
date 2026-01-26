import collections
import pathlib
import json
import pygsti
import numpy as np

import pygsti.circuits as pc
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYCNOT
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec as QPS
from qiskit import QuantumCircuit


class PyGSTiExperiment(object):
    def __init__(self, filepath, n_qubits, experiment_name):
        self.n_qubits = n_qubits
        self.experiment_name = experiment_name
        self.filepath = pathlib.Path(filepath)
        self.qasm_filepath = self.filepath / "qasm"

    def _pygsti_circuits_to_qasm_files(self):
        self.dataset_path = self.filepath / "data" / "dataset.txt"
        self.qasm_files.mkdir(parents=True, exist_ok=True)
        self.gatename_conversions = {
            "Gxpi2": ["rx(1.5707963267948966)"],
            "Gypi2": ["ry(1.5707963267948966)"],
            "Gcnot": ["cx"],
        }
        ds = pygsti.io.read_dataset(
            self.dataset_path,
            collision_action="aggregate",
            record_zero_counts=True,
            ignore_zero_count_lines=False,
        )
        print("Writing circuits to qasm files...")
        for line, circ in enumerate(ds):
            title = f"circ{line}.qasm"
            with open(self.qasm_files / title, "w+") as text_file:
                text_file.write(
                    circ.convert_to_openqasm(
                        gatename_conversion=self.gatename_conversions
                    )
                )
        print("Done.")

    def _qasm_to_qiskit_circuit_list(self):
        qasm_files = self.qasm_files.glob("**/*")
        qcs = {}
        for file in qasm_files:
            index = int(file.parts[-1][4:-5])
            qc = QuantumCircuit().from_qasm_file(file)
            qcs[index] = qc
        o_qcs = collections.OrderedDict(sorted(qcs.items()))
        # print(o_qcs)
        return list(o_qcs.values())

    # def generate_circuits(self):
    #     self.circuits = super().generate_circuits()
    #     return self.circuits

    def run_circuits(self, circuit_submitter, shots, skip_asking = False):
        self.submitter = circuit_submitter
        self.submitter.submit_circuits(shots=shots, qasm_strs=[c.qasm() for c in self.circuits], skip_asking=skip_asking)
        self.counts_list = self.submitter.retrieve_counts()
        print(self.counts_list)
        print(len(self.counts_list))
        return self.counts_list

    def write_counts_to_dataset(self):
        self.ds = pygsti.io.read_dataset(
            self.filepath / "data" / "dataset.txt",
            collision_action="aggregate",
            record_zero_counts=True,
            ignore_zero_count_lines=False,
        )

        new_ds = pygsti.data.DataSet(
            outcome_labels=[
                format(b, f"0{self.n_qubits}b") for b in range((self.n_qubits**2) - 1)
            ]
        )
        
        for counts, key in zip(self.counts_list, self.ds.keys()):
            new_ds[key] = counts
        pygsti.io.write_dataset(self.filepath / "data" / "dataset.txt", new_ds)

    def _dict_for_json(self):
        raise NotImplementedError

    def _to_json(self):
        json_object = json.dumps(self._dict_for_json(), indent=2)
        with open(self.filepath / "expdata.json", "w") as outfile:
            outfile.write(json_object)
