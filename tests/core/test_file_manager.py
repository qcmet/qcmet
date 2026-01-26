"""test_file_manager.py."""

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qcmet.core import FileManager


@pytest.fixture
def fixed_run_id():
    """Deterministic run_id for reproducible paths in tests."""
    return "20250101_000000"


@pytest.fixture
def fm(tmp_path, fixed_run_id):
    """FileManager with timestamped folder."""
    return FileManager(
        benchmark_name="bench",
        base_path=tmp_path,
        create_timestamp_folder=True,
        run_id=fixed_run_id,
    )


@pytest.fixture
def fm_no_ts(tmp_path):
    """FileManager without timestamped folder."""
    return FileManager(
        benchmark_name="bench",
        base_path=tmp_path,
        create_timestamp_folder=False,
    )


def test_directory_structure_created(fm, fixed_run_id):
    """Test creation of dir structure."""
    base = fm.base_path
    run_path = base / f"bench_{fixed_run_id}"
    assert run_path.exists() and run_path.is_dir()

    # Check expected subdirectories
    for sub in ["results", "data", "results/plots", "config", "logs", "intermediate"]:
        p = run_path / sub
        assert p.exists() and p.is_dir()


def test_directory_structure_without_timestamp(fm_no_ts):
    """Test dir structure when not using timestamp."""
    run_path = fm_no_ts.base_path / "bench"
    assert run_path.exists() and run_path.is_dir()

    # Check expected subdirectories
    for sub in ["results", "data", "results/plots", "config", "logs", "intermediate"]:
        p = run_path / sub
        assert p.exists() and p.is_dir()


def test_get_paths(fm, fixed_run_id):
    """Test returned paths are correct."""
    run_path = fm.base_path / f"bench_{fixed_run_id}"
    assert fm.get_results_path() == run_path / "results"
    assert fm.get_data_path() == run_path / "data"
    assert fm.get_plots_path() == run_path / "results" / "plots"
    assert fm.get_config_path() == run_path / "config"
    assert fm.get_logs_path() == run_path / "logs"
    assert fm.get_intermediate_path() == run_path / "intermediate"


def test_save_json_to_known_subfolders(fm):
    """Test json saving works."""
    data = {"a": 1}
    # results
    p1 = fm.save_json(data, "res_file", "results")
    assert p1.exists() and p1.suffix == ".json"
    # data
    p2 = fm.save_json(data, "data_file", "data")
    assert p2.exists() and p2.suffix == ".json"
    # plots (JSON here is allowed and should go under plots; though unusual)
    p3 = fm.save_json(data, "plot_file", "plots")
    assert p3.exists() and p3.suffix == ".json"
    # config
    p4 = fm.save_json(data, "cfg_file", "config")
    assert p4.exists() and p4.suffix == ".json"
    # logs
    p5 = fm.save_json(data, "log_file", "logs")
    assert p5.exists() and p5.suffix == ".json"
    # intermediate
    p6 = fm.save_json(data, "int_file", "intermediate")
    assert p6.exists() and p6.suffix == ".json"


def test_save_json_to_custom_subfolder(fm):
    """Test saving json to custom folder."""
    data = {"a": 42}
    p = fm.save_json(data, "custom_file", "custom_subfolder")
    assert p.exists()
    assert p.parent.name == "custom_subfolder"
    with p.open("r") as f:
        loaded = json.load(f)
    assert loaded == data


def test_make_json_serializable_scalar_types(fm, tmp_path):
    """Test direct serialization of various types."""
    dt = datetime(2020, 1, 1, 12, 0, 0)
    arr = np.array([1, 2, 3])
    np_int = np.int32(7)
    np_float = np.float64(3.14159)
    cx = 1 + 2j
    path = tmp_path / "foo.txt"

    assert fm._make_json_serializable(dt) == dt.isoformat()
    assert fm._make_json_serializable(arr) == [1, 2, 3]
    assert fm._make_json_serializable(np_int) == 7
    assert fm._make_json_serializable(np_float) == float(np_float)
    assert fm._make_json_serializable(cx) == {"real": 1.0, "imag": 2.0}
    assert fm._make_json_serializable(path) == str(path)

def test_save_json_mixed_types(fm, tmp_path):
    """Test json serialization works."""
    data = {
        "string": "hello",
        "int": 3,
        "float": 2.5,
        "bool": True,
        "none": None,
        "datetime": datetime(2020, 1, 1, 0, 0, 0),
        "np_array": np.array([1, 2, 3, 4]),
        "np_int": np.int64(42),
        "np_float": np.float64(0.5),
        "complex": 1 + 2j,
        "path": tmp_path / "bar.bin",
        "list": [1, 2, 3],
        "dict": {"nested": "ok"},
    }
    p = fm.save_json(data, "mixed_types", "results")
    assert p.exists()

    with p.open("r") as f:
        loaded = json.load(f)
        assert isinstance(loaded["datetime"], str)
        assert loaded["np_array"] == [1, 2, 3, 4]
        assert loaded["np_int"] == 42
        assert loaded["np_float"] == float(np.float64(0.5))
        assert loaded["complex"] == {"real": 1.0, "imag": 2.0}
        assert isinstance(loaded["path"], str)
        assert loaded["list"] == [1, 2, 3]
        assert loaded["dict"] == {"nested": "ok"}


def test_save_plot_creates_file(fm):
    """Test save plot."""
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    p = fm.save_plot(fig, "my_plot", "png")
    assert p.exists()
    assert p.suffix == ".png"
   
