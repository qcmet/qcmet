"""test_gate_set_tomography.py.

Unit tests for GST pygsti qcmet wrapper.
"""

from pathlib import Path

import numpy as np
import pygsti
import pytest
from pygsti.modelpacks import smq1Q_XY

import qcmet as qcm


@pytest.fixture
def gst_save_to_tmp_folder(tmp_path):
    """Set up basic gst fixture."""
    gst = qcm.GST(
        smq1Q_XY.create_gst_experiment_design(max_max_length=2), save_path=tmp_path
    )
    return gst


@pytest.fixture(scope="module")
def edesign():
    """Small 1-qubit experiment design for quick tests."""
    return smq1Q_XY.create_gst_experiment_design(max_max_length=2)


@pytest.fixture()
def gst_core_tmp(tmp_path, edesign):
    """GST configured for the core protocol (GateSetTomography)."""
    return qcm.GST(edesign, save_path=tmp_path, use_standard_gst=False)


@pytest.fixture()
def gst_standard_tmp(tmp_path, edesign):
    """GST configured for StandardGST (standard practice)."""
    return qcm.GST(edesign, save_path=tmp_path, use_standard_gst=True)


def _format_outcome_label(outcome) -> str:
    """Convert pyGSTi outcome labels to flat bitstrings.

    Examples:
      '0'         -> '0'
      ('0',)      -> '0'
      ('0','1')   -> '01'
      0           -> '0' (if you ever see ints; uncommon in pyGSTi)

    """
    if isinstance(outcome, str):
        return outcome
    if isinstance(outcome, (tuple, list)):
        # Common case: ('0',) or ('1',)
        if len(outcome) == 1 and isinstance(outcome[0], str):
            return outcome[0]
        return "".join(str(x) for x in outcome)
    return str(outcome)


def ds_to_qiskit_counts_list(ds, num_qubits: int):
    """Convert a pyGSTi DataSet into a list of Qiskit-style Counts dicts.

    Ensures integer counts and fills in any missing outcomes with 0.
    """
    counts_list = []
    # Preserve circuit key order
    for key in ds.keys():
        row_counts: dict[str, int] = {}
        for entry in ds[key]:
            outcome, *count = entry
            # print(np.array(count))
            count = int(count[1])
            label = _format_outcome_label(outcome)
            row_counts[label] = int(round(count))

        # Ensure a full computational-basis alphabet {0..2^n-        # Ensure a full computational-basis alphabet {0..2^n-1}
        for b in range(2**num_qubits):
            label = format(b, f"0{num_qubits}b")
            row_counts.setdefault(label, 0)

        counts_list.append(row_counts)
    return counts_list


def simulate_counts_for_edesign(edesign, model, num_shots=50, seed=1234):
    """Create deterministic per-circuit counts using pyGSTi and convert to Qiskit-style Counts."""
    # Important: pass a *list of circuits* (robust across py    # Important: pass a *list of circuits* (robust across pyGSTi versions)
    circuit_list = list(edesign.all_circuits_needing_data)

    ds = pygsti.data.datasetconstruction.simulate_data(
        model,
        circuit_list,
        num_samples=num_shots,
        sample_error="multinomial",
        seed=seed,
        record_zero_counts=True,  # include zero outcomes for completeness
    )
    return ds_to_qiskit_counts_list(ds, len(edesign.qubit_labels))


def test_init_writes_protocol_tree_and_resolves_model(gst_core_tmp):
    """Init should create the design/data tree and resolve a pyGSTi Model."""
    path = Path(gst_core_tmp.config["experiment_design_path"])
    assert path.exists()

    ds_file = path / "data" / "dataset.txt"
    assert ds_file.exists(), "Skeleton dataset.txt must exist after init."

    assert hasattr(gst_core_tmp, "_ideal_model")
    assert isinstance(gst_core_tmp._ideal_model, pygsti.models.model.Model)


def test_generate_circuits_count_matches_design(gst_core_tmp):
    """Generated QASM circuits should match the number of design circuits."""
    gst_core_tmp.generate_circuits()
    assert isinstance(gst_core_tmp.circuits, list) and len(gst_core_tmp.circuits) > 0
    assert len(gst_core_tmp.circuits) == len(
        gst_core_tmp._experiment_design.all_circuits_needing_data
    )


def test_write_counts_sets_2powN_outcomes(gst_core_tmp):
    """Ensure DataSet uses computational-basis bitstrings of size 2**n."""
    gst_core_tmp.generate_circuits()
    counts_list = simulate_counts_for_edesign(
        gst_core_tmp._experiment_design, gst_core_tmp._ideal_model, num_shots=20, seed=1
    )
    gst_core_tmp.load_circuit_measurements(counts_list)
    gst_core_tmp._write_counts_to_dataset()

    ds = pygsti.io.read_dataset(
        Path(gst_core_tmp.config["experiment_design_path"]) / "data" / "dataset.txt",
        collision_action="aggregate",
        record_zero_counts=True,
        ignore_zero_count_lines=False,
    )
    n = len(gst_core_tmp.qubits)
    expected = {format(b, f"0{n}b") for b in range(2**n)}
    for key in ds.keys():
        # print(ds[key].ol)
        ds_keys = ds[key].to_dict().keys()
        ds_outcome_labels = {_format_outcome_label(i) for i in ds_keys}
        assert ds_outcome_labels == expected


def test_build_protocol_types(gst_core_tmp, gst_standard_tmp):
    """_build_protocol should return the correct protocol class."""
    init_core = pygsti.protocols.GSTInitialModel(gst_core_tmp._ideal_model)
    proto_core = gst_core_tmp._build_protocol(init_core)
    assert isinstance(proto_core, pygsti.protocols.GateSetTomography)

    init_std = pygsti.protocols.GSTInitialModel(gst_standard_tmp._ideal_model)
    proto_std = gst_standard_tmp._build_protocol(init_std)
    assert isinstance(proto_std, pygsti.protocols.StandardGST)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analyze_core_returns_metrics(gst_core_tmp):
    """Run core GST on deterministic counts and check metrics structure."""
    gst_core_tmp.generate_circuits()
    counts_list = simulate_counts_for_edesign(
        gst_core_tmp._experiment_design, gst_core_tmp._ideal_model, num_shots=30, seed=7
    )
    gst_core_tmp.load_circuit_measurements(counts_list)

    results = gst_core_tmp.analyze()
    assert isinstance(results, dict)
    assert "SPAM fidelity" in results

    # Gate fidelity keys exist and are floats in [0,1]
    gate_keys = [k for k in results if k.startswith("Process Fidelity Gate:")]
    assert gate_keys, "No gate fidelity metrics found."
    for k in gate_keys:
        v = results[k]
        v = (
            v[0] if isinstance(v, tuple) else v
        )  # class currently returns float, earlier used tuple
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    # SPAM metrics present and numeric
    spam = results["SPAM fidelity"]
    assert {"Fidelity of initial density matrix", "Measurement Fidelity"} <= set(
        spam.keys()
    )
    assert isinstance(spam["Fidelity of initial density matrix"], float)
    assert isinstance(spam["Measurement Fidelity"], float)


def test_gst_result(gst_save_to_tmp_folder):
    """End to end test of GST."""
    gst_save_to_tmp_folder.generate_circuits()
    noisy_sim = qcm.NoisySimulator(
        detuning_amount=0,
        error_1q=0,
        error_2q=0.01,
        overrotation_amount=0,
        num_qubits=1,
        t1=0,
        t2=0,
    )
    gst_save_to_tmp_folder.run(noisy_sim, num_shots=100)
    result = gst_save_to_tmp_folder.analyze()
    np.testing.assert_allclose(
        result["Process Fidelity Gate:('Gxpi2', 0)"], 0.99, rtol=0.011
    )
    np.testing.assert_allclose(
        result["Process Fidelity Gate:('Gypi2', 0)"], 0.99, rtol=0.011
    )
    np.testing.assert_allclose(
        result["SPAM fidelity"]["Fidelity of initial density matrix"], 1, rtol=0.011
    )
    np.testing.assert_allclose(
        result["SPAM fidelity"]["Measurement Fidelity"], 1, rtol=0.011
    )

    est_name = next(iter(gst_save_to_tmp_folder.gst_results.estimates.keys()))
    est = gst_save_to_tmp_folder.gst_results.estimates[est_name]
    selected = gst_save_to_tmp_folder._select_best_estimate_model(
        gst_save_to_tmp_folder.gst_results
    )

    # If 'stdgaugeopt' exists, it must be the selected model
    if "stdgaugeopt" in est.models:
        assert selected is est.models["stdgaugeopt"]
    else:
        assert isinstance(selected, pygsti.models.model.Model)


def test_temp_dir_cleanup_when_no_save_path(edesign):
    """__del__ should be guarded and not error when using a TemporaryDirectory."""
    gst = qcm.GST(edesign, use_standard_gst=False, save_path=None)
    tmp_path = Path(gst.config["experiment_design_path"])
    assert tmp_path.exists()
