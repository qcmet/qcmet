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

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
from _helpers.pygsti_experiment import *
from _helpers.circuit_submitter import CircuitSubmitter


class GST(PyGSTiExperiment):
    def __init__(
        self, filepath, n_qubits=2, maxLengths=None, circuit_reduction_method="fpr"
    ):
        super().__init__(filepath, n_qubits, "GST")
        self.qasm_files = pathlib.Path.joinpath(self.filepath, "data", "qasm")
        if maxLengths is None:
            self.maxLengths = [1, 2, 4, 8, 16]
        else:
            self.maxLengths = maxLengths
        if n_qubits == 1:
            self.model = smq1Q_XYI
        elif n_qubits == 2:
            self.model = smq2Q_XYCNOT

        self.circuit_reduction_method = circuit_reduction_method

    def generate_circuits(self):
        self._generate_gst_folder()
        self._pygsti_circuits_to_qasm_files()
        self.circuits = self._qasm_to_qiskit_circuit_list()
        return self.circuits

    def _generate_gst_folder(self):
        if self.circuit_reduction_method == "fpr":
            self._generate_gst_folder_with_fpr()
        elif self.circuit_reduction_method == "rfpr":
            self._generate_gst_folder_with_rfpr()
        elif (
            self.circuit_reduction_method == None
            or self.circuit_reduction_method.lower() == "none"
        ):
            self.exp_design = self.model.create_gst_experiment_design(
                max_max_length=max(self.maxLengths)
            )
            pygsti.io.write_empty_protocol_data(
                self.filepath, self.exp_design, clobber_ok=True
            )
        else:
            raise ValueError("Invalid circuit reduction method")

    def _generate_gst_folder_with_fpr(self):
        self.exp_design = self.model.create_gst_experiment_design(
            max_max_length=max(self.maxLengths), fpr=True
        )
        pygsti.io.write_empty_protocol_data(
            self.filepath, self.exp_design, clobber_ok=True
        )

    def _generate_gst_folder_with_rfpr(self):
        self.target_model = self.model.target_model()
        self.prep_fiducials = self.model.prep_fiducials()
        self.meas_fiducials = self.model.meas_fiducials()
        self.germs = self.model.germs()

        self.opLabels = list(self.target_model.operations.keys())
        print("Gate operation labels = ", self.opLabels)

        self.pfprStructs = pc.create_lsgst_circuit_lists(
            self.opLabels,
            self.prep_fiducials,
            self.meas_fiducials,
            self.germs,
            self.maxLengths,
            keep_fraction=0.125,
        )
        print("\nPer-germ FPR reduction")
        for L, strct in zip(self.maxLengths, self.pfprStructs):
            print("L=%d: %d operation sequences" % (L, len(strct)))

        self.pfprExperiments = pc.create_lsgst_circuits(
            self.opLabels,
            self.prep_fiducials,
            self.meas_fiducials,
            self.germs,
            self.maxLengths,
            keep_fraction=0.125,
        )
        print("\n%d experiments to run GST." % len(self.pfprExperiments))

        self.pspec = self.model.target_model().create_processor_spec()
        self.gst_design_pfpr = pygsti.protocols.GateSetTomographyDesign(
            self.pspec, self.pfprExperiments
        )

        pygsti.io.write_empty_protocol_data(
            str(self.filepath), self.gst_design_pfpr, clobber_ok=True
        )

    def _dict_for_json(self):
        return {
            "n_qubits": self.n_qubits,
            "maxLengths": self.maxLengths,
            "circuit_reduction_method": self.circuit_reduction_method,
        }


if __name__ == "__main__":
    n_qubits = 1
    maxLengths = [1, 2, 4, 8, 16]
    shots = 1000
    device_name = "noisy_sim"

    submitter = CircuitSubmitter(benchmark_name="gst", device_name=device_name)
    # Uncomment the following lines if you are using a noisy simulator and would like to change the noise model
    # noise_model = None
    # submitter.backend.device.noisy_sim.set_options(noise_model=noise_model)
    filepath = submitter.benchmark_path

    gst_experiment = GST(filepath=filepath, n_qubits=n_qubits, maxLengths=maxLengths, circuit_reduction_method="fpr")
    gst_experiment.generate_circuits()
    gst_experiment.run_circuits(submitter, shots=shots)
    gst_experiment.write_counts_to_dataset()