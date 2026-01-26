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

import os
import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np
import qiskit
from qiskit import Aer
from qiskit.providers.fake_provider import FakeKolkataV2
from qiskit.converters import circuit_to_dag

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent.parent.resolve()))
from _helpers.circuit_submitter import CircuitSubmitter

from tqdm.auto import tqdm
import uuid


def random_complex_matrix(n):
    return np.random.randn(n, n) + 1.0j * np.random.randn(n, n)


def haar_measure(n):
    """
    Random unitary from Haar measure with dim n*n
    arXiv:math-ph/0609050
    """
    rand_mat = random_complex_matrix(n)
    q, r = np.linalg.qr(rand_mat)
    d = np.diagonal(r)
    d_normed = d / np.absolute(d)
    return np.multiply(q, d_normed, q)


def find_active_qubits(circuit):

    dag = circuit_to_dag(circuit)
    active_qubits = [
        qubit.index for qubit in circuit.qubits if qubit not in dag.idle_wires()
    ]

    return active_qubits


def apply_su4_layer(qc, num_qubits):
    even_num_qubits = int(np.floor(num_qubits / 2) * 2)
    for qubit in range(0, even_num_qubits, 2):
        random_su4_gate = haar_measure(4)
        qc.append(
            qiskit.circuit.library.UnitaryGate(random_su4_gate), [qubit, qubit + 1]
        )


def apply_swap_layer(qc, num_qubits):
    permutation_list = np.random.permutation(num_qubits)
    init_order = np.arange(num_qubits)
    for i in range(num_qubits):
        desired_qubit_at_i = permutation_list[i]
        loc_current_to_swap = i
        loc_desired_to_swap = np.where(init_order == desired_qubit_at_i)[0][0]
        if loc_current_to_swap != loc_desired_to_swap:
            qc.swap(loc_current_to_swap, loc_desired_to_swap)
            init_order[loc_current_to_swap], init_order[loc_desired_to_swap] = (
                init_order[loc_desired_to_swap],
                init_order[loc_current_to_swap],
            )


def apply_qv_layer(qc, num_qubits):
    apply_swap_layer(qc, num_qubits)
    apply_su4_layer(qc, num_qubits)


def qiskit_counts_to_probs(counts):
    counts_array = np.asarray([(int(k, 2), v) for k, v in counts.items()])
    sorted_counts_array = counts_array[counts_array[:, 0].argsort()]
    prob_array = sorted_counts_array[:, 1] / np.sum(sorted_counts_array[:, 1])
    return prob_array


def qiskit_counts_to_sorted_array(counts):
    counts_array = np.asarray([(int(k, 2), v) for k, v in counts.items()])
    sorted_counts_array = counts_array[counts_array[:, 0].argsort()]
    prob_array = sorted_counts_array[:, 1] / np.sum(sorted_counts_array[:, 1])
    probs_ascending_order = np.argsort(prob_array)
    sorted_probs = prob_array[probs_ascending_order]
    sorted_bit_strings = sorted_counts_array[:, 0][probs_ascending_order]
    return sorted_probs, sorted_bit_strings


def get_heavy_outputs(num_qubits, sorted_output_array):
    output_strings = np.asarray(
        [f"{x:0{num_qubits}b}" for x in range(sorted_output_array.shape[0])]
    )
    heavy_output_strings = output_strings[
        np.where(sorted_output_array > np.median(sorted_output_array))
    ]
    prob_above_median = sorted_output_array[
        sorted_output_array > np.median(sorted_output_array)
    ]
    heavy_output_prob = np.sum(prob_above_median)
    return heavy_output_strings, prob_above_median, heavy_output_prob


def qv_circuit(num_qubits=5, print_circuit=True):
    qc = qiskit.QuantumCircuit(num_qubits)
    for _ in range(num_qubits):
        apply_qv_layer(qc, num_qubits)
    qc.measure_all()
    if print_circuit:
        print(qc)

    return qc


def get_counts(
    qc,
    simulator,
    shots=5000,
    angle=None,
    optimization_level=1,
    qc_id=None,
    sim_name=None,
):
    circ = qiskit.transpile(
        qc,
        simulator,
        optimization_level=optimization_level,
        basis_gates=["rx", "ry", "rz", "cx"],
    )

    num_qubits = qc.num_qubits

    CIRC_DATA_PATH = os.path.join(PATH, sim_name, "circ_data")
    CIRC_DIAGRAM_PATH = os.path.join(PATH, sim_name, "circ_diagrams")
    os.makedirs(CIRC_DATA_PATH, exist_ok=True)
    os.makedirs(CIRC_DIAGRAM_PATH, exist_ok=True)
    if qc_id:
        # print('Depth: ', circ.depth())
        num_cx_gates = 0
        for i, op in enumerate(circ.data):
            if type(op[0]) is qiskit.circuit.library.standard_gates.x.CXGate:
                num_cx_gates += 1
        with open(
            os.path.join(CIRC_DATA_PATH, f"circ_data_vol_{num_qubits}.txt"), "a+"
        ) as f:
            f.write(
                f"{num_qubits},{optimization_level},{num_cx_gates},{circ.depth()},{str(qc_id)}\n"
            )
        circ.qasm(filename=os.path.join(CIRC_DIAGRAM_PATH, str(qc_id)))

    result = simulator.run(circ, shots=shots).result()
    counts = result.get_counts(circ)
    return counts


def get_ideal_counts(qc_ideal, optimization_level=1):
    qc_id = uuid.uuid4()
    return get_counts(
        qc_ideal,
        ideal_simulator,
        optimization_level=optimization_level,
        qc_id=f"ideal{str(qc_id)}",
        sim_name="ideal",
    )


def get_noisy_and_ideal_counts(qc_ideal, qc_noisy, noisy_sim, optimization_level=1):
    qc_id = uuid.uuid4()
    return get_counts(
        qc_ideal,
        ideal_simulator,
        optimization_level=optimization_level,
        qc_id=f"ideal{str(qc_id)}",
        sim_name=sim_name,
    ), get_counts(
        qc_noisy,
        noisy_sim,
        optimization_level=optimization_level,
        qc_id=f"noisy{str(qc_id)}",
        sim_name=sim_name,
    )


def print_heavy_outputs(heavy_outputs_strings, heavy_output_probs, prob_heavy_output):
    print("State   Prob")
    for i, j in zip(heavy_outputs_strings, heavy_output_probs):
        print(i, "  ", j)
    print("Heavy output probability: ", prob_heavy_output)


def plot_heavy_output_distribution(
    data, noisy_sim_name, title, num_trials, save=False, path=None, show_plot=False
):
    # plt.figure()
    probs_mean_noisy = np.mean(data)
    # num_trials = data.shape[0]
    stds_noisy = np.sqrt(probs_mean_noisy * (1 - probs_mean_noisy) / num_trials)

    plt.hist(data)
    plt.axvline(probs_mean_noisy, c="b", label="Mean")
    plt.axvline(
        probs_mean_noisy - 2 * stds_noisy, c="r", linestyle="--", label=r"2$\sigma$"
    )
    plt.axvline(2 / 3, c="black", label="2/3")
    plt.title(title)
    ymin, ymax = plt.gca().get_ylim()
    plt.xlim((0.5, 1.0))
    plt.ylabel("Occurences")
    plt.xlabel("Heavy Output Probability")
    plt.legend()
    if save:
        if path is None:
            path = os.path.join(PATH, noisy_sim_name, "plots")
        os.makedirs(path, exist_ok=True)
        plt.savefig(
            os.path.join(path, str(title) + ".png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
    if show_plot:
        plt.show()
    # plt.clf()


def plot_average_heavy_output(
    ideal_output_array_mean,
    noisy_output_array_mean,
    noisy_sim_name,
    num_trials,
    title,
    save=False,
    path=None,
    show_plot=False,
):
    # plt.figure()
    plt.rcParams.update({"font.size": 14})
    num_qubits = np.arange(2, len(ideal_output_array_mean) + 2)
    print(num_qubits)
    print(ideal_output_array_mean)
    print(noisy_output_array_mean)
    plt.scatter(num_qubits, ideal_output_array_mean, label="ideal sim", marker="^")
    plt.scatter(num_qubits, noisy_output_array_mean, label="noisy sim")

    stds_noisy = np.asarray([np.sqrt(i * (1 - i) / num_trials) for i in noisy_output_array_mean])
    plt.errorbar(
        num_qubits,
        noisy_output_array_mean,
        2*stds_noisy,
        ls="none",
        ecolor="#ff7f0e",
        capsize=1.5,
        alpha=0.6,
        elinewidth=1,
    )
    plt.axhline(
        0.85,
        label="ideal heavy output probability",
        c="green",
        linestyle="--",
    )
    plt.axhline(
        2 / 3,
        label="threshold",
        c="r",
        linestyle="--",
    )
    plt.xlabel(r"width/depth of circuit, $m=d$" + f" on {noisy_sim_name}")
    plt.ylabel(r"est. heavy output probability, $\hat{h}$")
    plt.legend(fontsize=12)
    plt.xticks(num_qubits)
    plt.title(title)

    plt.ylim((0.5, 1.0))
    if save:
        if path is None:
            path = os.path.join(PATH, noisy_sim_name, "plots")
        os.makedirs(path, exist_ok=True)
        plt.savefig(
            os.path.join(
                path,
                title + f"{noisy_sim_name}_average_heavy_output_noisy_vs_ideal.png",
            ),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
    if show_plot:
        plt.show()
    # plt.clf()


def qv_trial(vol, noisy_sim, optimization_level=1):
    qc_ideal = qv_circuit(vol, print_circuit=False)
    qc_noisy = qc_ideal.copy()
    ideal_counts, noisy_counts = get_noisy_and_ideal_counts(
        qc_ideal, qc_noisy, noisy_sim, optimization_level=optimization_level
    )

    sorted_ideal_prob_array, bitsrings = qiskit_counts_to_sorted_array(ideal_counts)
    noisy_prob_array = qiskit_counts_to_probs(noisy_counts)
    sorted_noisy_prob_array = noisy_prob_array[bitsrings]
    (
        ideal_heavy_outputs_strings,
        ideal_heavy_output_probs,
        ideal_prob_heavy_output,
    ) = get_heavy_outputs(vol, sorted_ideal_prob_array)
    (
        noisy_heavy_outputs_strings,
        noisy_heavy_output_probs,
        noisy_prob_heavy_output,
    ) = get_heavy_outputs(vol, sorted_noisy_prob_array)
    return ideal_prob_heavy_output, noisy_prob_heavy_output


def generate_qv_circuits(num_qubit_list, num_trials=100, optimization_level=1):
    qc_list = {}
    for num_qubits in num_qubit_list:
        qc_list[num_qubits] = []

        submitter_noiseless = CircuitSubmitter("quantum_volume", "noiseless_sim")
        for _ in range(num_trials):
            qc = qv_circuit(num_qubits=num_qubits, print_circuit=False)
            qc = qiskit.transpile(qc, basis_gates=["rx", "ry", "rz", "cx"])
            submitter_noiseless.submit_circuits(
                shots=1000, qasm_strs=[qc.qasm()], skip_asking=True, print_summary=False
            )
            ideal_counts = submitter_noiseless.retrieve_counts(wait=True, print_timestamp_when_done=False)[0]
            sorted_ideal_prob_array, bitsrings = qiskit_counts_to_sorted_array(
                ideal_counts
            )
            qc_data = {
                "qc": qc,
                "sorted_ideal_prob_array": sorted_ideal_prob_array,
                "bitstrings": bitsrings,
            }
            qc_list[num_qubits].append(qc_data)

    return qc_list


def run_qv_test(
    num_qubits_list=[2, 3], num_trials=100, optimization_level=1, circuit_submitter=None
):
    qc_list = generate_qv_circuits(
        num_qubits_list, num_trials=num_trials, optimization_level=optimization_level
    )
    if circuit_submitter is None:
        circuit_submitter = CircuitSubmitter("quantum_volume", "noisy_sim")
    for n_qubits in num_qubits_list:
        circuits = [trial["qc"].qasm() for trial in qc_list[n_qubits]]
        circuit_submitter.submit_circuits(
            shots=1000, qasm_strs=circuits, skip_asking=True, print_summary=False
        )
        all_counts = circuit_submitter.retrieve_counts(wait=True)
        noisy_prob_arrays = [qiskit_counts_to_probs(counts) for counts in all_counts]
        for trial, noisy_probs in zip(qc_list[n_qubits], noisy_prob_arrays):
            sorted_noisy_prob_array = noisy_probs[trial["bitstrings"]]
            (
                noisy_heavy_outputs_strings,
                noisy_heavy_output_probs,
                noisy_prob_heavy_output,
            ) = get_heavy_outputs(n_qubits, sorted_noisy_prob_array)
            (
                ideal_heavy_outputs_strings,
                ideal_heavy_output_probs,
                ideal_prob_heavy_output,
            ) = get_heavy_outputs(n_qubits, trial["sorted_ideal_prob_array"])
            trial["noisy_prob_heavy_output"] = noisy_prob_heavy_output
            trial["ideal_prob_heavy_output"] = ideal_prob_heavy_output
            trial["noisy_heavy_output_probs"] = noisy_heavy_output_probs
            trial["ideal_heavy_output_probs"] = ideal_heavy_output_probs
    return qc_list


if __name__ == "__main__":
    from tqdm import tqdm
    device_name = "noisy_sim"
    submitter = CircuitSubmitter(benchmark_name="quantum_volume", device_name=device_name)
    filepath = submitter.benchmark_path

    num_qubits_list = [2, 3, 4, 5, 6, 7,]
    num_trials = 200
    optimization_level = 1

    # Uncomment the following lines if you are using a noisy simulator and would like to change the noise model
    # from qiskit_aer.noise import NoiseModel
    # backend = FakeKolkataV2()
    # # noise_model = None
    # submitter.backend.noise_model = backend._get_noise_model_from_backend_v2()

    sim_name = submitter.device_name

    qc_list = run_qv_test(
        num_qubits_list=num_qubits_list,
        num_trials=num_trials,
        optimization_level=optimization_level,
        circuit_submitter=submitter,
    )

    ideal_heavy_outputs_dict = {}
    noisy_heavy_outputs_dict = {}
    mean_ideal_heavy_outputs = []
    mean_noisy_heavy_outputs = []

    for n_qubits in tqdm(num_qubits_list):
        print(f'Number of qubits: {n_qubits}: \n')

        ideal_heavy_outputs = np.asarray(
                    [trial["ideal_prob_heavy_output"] for trial in qc_list[n_qubits]]
                )
        ideal_heavy_outputs_dict[n_qubits] = ideal_heavy_outputs
        mean_ideal_heavy_outputs.append(np.mean(ideal_heavy_outputs))
        
        noisy_heavy_outputs = np.asarray(
                    [trial["noisy_prob_heavy_output"] for trial in qc_list[n_qubits]]
                )
        noisy_heavy_outputs_dict[n_qubits] = noisy_heavy_outputs
        mean_noisy_heavy_outputs.append(np.mean(noisy_heavy_outputs))
        

        plot_heavy_output_distribution(
            np.asarray(
                [
                    np.sum(trial["noisy_heavy_output_probs"])
                    for trial in qc_list[n_qubits]
                ]
            ),
            sim_name,
            f"Noisy simulation m={n_qubits} for {sim_name}  with opt_level {optimization_level}",
            num_trials,
            save=True,
            path=filepath,
            show_plot=True,
        )
        plot_heavy_output_distribution(
            np.asarray(
                [
                    np.sum(trial["ideal_heavy_output_probs"])
                    for trial in qc_list[n_qubits]
                ]
            ),
            sim_name,
            f"Ideal simulation m={n_qubits} for {sim_name} with opt_level {optimization_level}",
            num_trials,
            save=True,
            path=filepath,
        )
    plot_average_heavy_output(
        mean_ideal_heavy_outputs,
        mean_noisy_heavy_outputs,
        sim_name,
        num_trials,
        f"{num_trials} trials for optimisation level {optimization_level}",
        save=True,
        path=filepath,
        show_plot=True,
    )


    with open(filepath + '/results.pkl', 'wb') as f:
        pickle.dump([ideal_heavy_outputs_dict, noisy_heavy_outputs_dict], f)