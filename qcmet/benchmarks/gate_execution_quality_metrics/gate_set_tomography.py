"""QCMet wrapper for pyGSTi Gate Set Tomography (GST).

This module provides a thin QCMet-friendly wrapper that:
- Writes a pyGSTi experiment design and dataset skeleton to disk.
- Runs GST via either `StandardGST` (standard practice) or the core
  `GateSetTomography` protocol (optionally customizable).
- Computes concise gate and SPAM quality metrics for downstream reporting.

Notes:
    For highly customized GST (objective functions, constraints, gauge
    metrics/groups, leakage-aware models, parallel MPI, custom reports),
    please interact with **pyGSTi directly**. This wrapper stays minimal
    to reduce maintenance when pyGSTi evolves.

"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from qcmet.core import FileManager

from pathlib import Path
from tempfile import TemporaryDirectory

import pygsti
from pygsti.io import (
    read_data_from_dir,
    read_dataset,
    write_dataset,
    write_empty_protocol_data,
)
from pygsti.report import reportables as rptbl
from qiskit.qasm2 import loads

from qcmet.benchmarks import BaseBenchmark

# Type aliases for readability
Model = pygsti.models.model.Model
QPS = pygsti.processors.QubitProcessorSpec
GSTDesign = pygsti.protocols.GateSetTomographyDesign
GSTInitialModel = pygsti.protocols.GSTInitialModel
GateSetTomography = pygsti.protocols.GateSetTomography
StandardGST = pygsti.protocols.StandardGST


class GST(BaseBenchmark):
    """Run Gate Set Tomography (GST) with pyGSTi in a minimal QCMet-friendly way.

    Args:
        gst_experiment_design (GSTDesign): A pyGSTi experiment design that
            defines GST circuits, qubit labels, and a processor spec.
        save_path (str | Path | FileManager | None, optional): Location to
            store design, dataset, and results. If `None`, a temporary
            directory is used and cleaned up on destruction.
        target_model (Model | None, optional): Explicit pyGSTi model to use
            as the initial/target model. Must be compatible with the design.
        target_model_factory (Callable[[QPS], Model] | None, optional):
            Callback that builds a model from the design `processor_spec`.
        model_pack (object | None, optional): A `pygsti.modelpacks` module
            (e.g., `smq1Q_XY`) providing `target_model(parameterization)`.
        parameterization (str, optional): Parameterization passed to
            `model_pack.target_model(...)` (e.g., `'TP'`, `'CPTP'`, `'full'`).
            Used only when `model_pack` is supplied.
        use_standard_gst (bool, optional): If `True`, run `StandardGST`
            (standard-practice long-sequence GST + default gauge optimization).
            If `False`, run the core `GateSetTomography` protocol.
        gst_gaugeopt_suite (object | None, optional): Core-path only. A
            `GSTGaugeOptSuite` or dict of directives describing gauge
            optimization passes. If `None`, core path does no gauge opt.
        gst_objfn_builders (object | None, optional): Core-path only. A
            `GSTObjFnBuilders` specifying long-sequence objective functions
            (e.g., χ² or log-likelihood).
        gst_optimizer (object | str | None, optional): Core-path only.
            A pyGSTi optimizer instance or `'auto'` for numerical fitting.
        gst_badfit_options (object | None, optional): Core-path only.
            A `GSTBadFitOptions` defining remediation steps for poor fits.
        gst_verbosity (int, optional): Core-path only. pyGSTi runner verbosity.

    Notes:
        This module provides a thin QCMet-friendly wrapper that:
        - Writes a pyGSTi experiment design and dataset skeleton to disk.
        - Runs GST via either `StandardGST` (standard practice) or the core
        `GateSetTomography` protocol (optionally customizable).
        - Computes concise gate and SPAM quality metrics for downstream reporting.


        - It exposes two analysis modes:
            - **Standard practice GST** (`StandardGST`): a canonical orchestration of
                long-sequence GST that includes iterative objective functions and a built-in
                gauge-optimization suite. This is the quickest path to a gauge-optimized
                estimate using widely adopted defaults.
            - **Core GST** (`GateSetTomography`): the low-level protocol performing the
                actual parameter estimation. This path is intentionally lean and can be
                configured via optional kwargs (gauge-opt suite, objective builders,
                optimizer, and “bad fit” options). If you leave them `None`, the core path
                runs with minimal assumptions and skips gauge optimization.

        - After analysis, it computes concise **quality metrics**:
            - Per-gate **process fidelity** (ideal vs. fitted model).
            - Optional per-gate **(half) diamond norm** (requires `cvxpy`).
            - **SPAM** metrics:
                - Fidelity of the prepared |0⟩ state (as Hilbert–Schmidt vectors).
                - Aggregate measurement fidelity from POVM entanglement infidelity.

        Design
        ------
        This wrapper aims to be as basic as possible while still providing the hooks
        for more advanced usage:
        - All heavy lifting (experiment design, objective functions, fitting, gauge
        optimization) is handled by pyGSTi.
        - The wrapper only standardizes I/O (dataset writing) and provides a set
        of result metrics .

        Advanced usage — please use pyGSTi directly
        -------------------------------------------
        For research-level customization (new objective functions, bespoke constraints,
        novel gauge groups/metrics, leakage-aware modeling, etc.), it is advisable
        to interact with pyGSTi directly:

        - Protocols: `pygsti.protocols.gst` (e.g., `GateSetTomography`, `StandardGST`,
        - `GSTObjFnBuilders`, `GSTGaugeOptSuite`).
        - Algorithms: `pygsti.algorithms.core` and `pygsti.algorithms.gaugeopt`.
        - Models: `pygsti.models.*` (explicit/crosstalk/cloud-noise).
        - Data & design: `pygsti.io`, `pygsti.protocols.ExperimentDesign`.

        Data flow
        ---------
        1) You pass a `GateSetTomographyDesign` that encodes the circuits, qubit labels,
        and processor spec.
        2) QCMet executes those circuits using the `run()` method; and writes counts
        into pyGSTi the pyGSTi `dataset.txt`.
        3) `analyze()` runs the selected protocol on the design+dataset (read from disk),
        selects the best estimate (preferring `'stdgaugeopt'`), then computes metrics.

        Target/ideal model
        ------------------
        GST is model-based. This class resolves the target/initial model via:
        (1) `target_model` (explicit), or
        (2) `target_model_factory(processor_spec)`, or
        (3) `model_pack.target_model(parameterization)`, or
        (4) `pygsti.models.create_explicit_model(processor_spec)` (fallback).

        I/O & storage
        -------------
        • If `save_path` is a QCMet `FileManager` or filesystem path, everything is written
        there; otherwise a temporary directory is used (and cleaned up on destruction).
        • Outcome labels are set to **2**^**n** bitstrings (e.g., `'00'`, `'01'`) to match
        standard qubit POVMs. Adjust if your design uses a different outcome alphabet.

        Performance & parallelization
        -----------------------------
        pyGSTi supports MPI/multiprocessing for large designs. This wrapper does not
        configure parallelism — please use pyGSTi directly to tune performance.
        One may save to disk and use the run function, but leave the analysis
        to bare pyGSTi code.

        Assumptions
        -----------
        • The default POVM label used in metrics is `"Mdefault"`.
        • Circuits are exported as OpenQASM 2 via `convert_to_openqasm()`; map/compile
        upstream if your backend expects other intermediate representations.

    """

    def __init__(
        self,
        gst_experiment_design: GSTDesign,
        save_path: str | Path | FileManager | None = None,
        *,
        target_model: Optional[Model] = None,
        target_model_factory: Optional[Callable[[QPS], Model]] = None,
        model_pack: Optional[object] = None,
        parameterization: str = "CPTP",
        use_standard_gst: bool = False,
        gst_gaugeopt_suite: object | None = None,
        gst_objfn_builders: object | None = None,
        gst_optimizer: object | str | None = None,
        gst_badfit_options: object | None = None,
        gst_verbosity: int = 0,
    ):
        """Initialize the GST analyzer and serialize the experiment design.

        Args:
            gst_experiment_design (GSTDesign): pyGSTi experiment design
                specifying circuits, qubit labels, and processor spec.
            save_path (str | Path | FileManager | None, optional): Storage
                location for design/data/results. If `None`, a temp dir is used.
            target_model (Model | None, optional): Explicit initial/target model.
            target_model_factory (Callable[[QPS], Model] | None, optional):
                Factory to build a model from the design's `processor_spec`.
            model_pack (object | None, optional): Model pack providing a
                `target_model(parameterization)` function.
            parameterization (str, optional): Parameterization for the model
                pack's target model (`'TP'`, `'CPTP'`, `'full'`).
            use_standard_gst (bool, optional): Select `StandardGST` (True) or
                basic `GateSetTomography` (False).
            gst_gaugeopt_suite (object | None, optional): basic GST gauge-opt suite
                or directives. If `None`, no gauge optimization is performed.
            gst_objfn_builders (object | None, optional): basic GST objective
                builders for long-sequence GST.
            gst_optimizer (object | str | None, optional): basic GST optimizer
                instance or `'auto'`.
            gst_badfit_options (object | None, optional): basic GST remediation
                options for poor fits.
            gst_verbosity (int, optional): basic GST runner verbosity.

        Notes:
            - Immediately writes an empty pyGSTi protocol data tree to
              `self.config["experiment_design_path"]` so analysis can read it.
            - Resolves the initial/target model via precedence:
              `target_model` -> `target_model_factory` -> `model_pack` ->
              `create_explicit_model(processor_spec)`.

        """
        qubit_labels = list(gst_experiment_design.qubit_labels)
        super().__init__("GST", qubits=qubit_labels, save_path=save_path)

        self._experiment_design: GSTDesign = gst_experiment_design
        self._use_standard_gst = use_standard_gst

        # Core-GST knobs (used only when not using StandardGST)
        self._gst_gaugeopt_suite = gst_gaugeopt_suite
        self._gst_objfn_builders = gst_objfn_builders
        self._gst_optimizer = gst_optimizer
        self._gst_badfit_options = gst_badfit_options
        self._gst_verbosity = gst_verbosity

        # Serialize design to disk
        self._init_storage_and_write_design()

        # Resolve a compatible target/ideal model by precedence
        self._ideal_model: Model = self._resolve_target_model(
            target_model=target_model,
            target_model_factory=target_model_factory,
            model_pack=model_pack,
            parameterization=parameterization,
        )

    def _init_storage_and_write_design(self) -> None:
        """Initialize storage paths and write pyGSTi's empty protocol data."""
        if self.file_manager is not None:
            self.config["experiment_design_path"] = (
                self.file_manager.get_intermediate_path()
            )
            self.config["experiment_design"] = (
                f"Experiment data saved at {self.file_manager.run_path}"
            )
        else:
            self._temp_dir = TemporaryDirectory()
            self.config["experiment_design_path"] = Path(self._temp_dir.name)
            self.config["experiment_design"] = (
                f"Experiment data saved at {self.config['experiment_design_path']}"
            )

        write_empty_protocol_data(
            self.config["experiment_design_path"],
            self._experiment_design,
            clobber_ok=True,
        )

    def _resolve_target_model(
        self,
        *,
        target_model: Optional[Model],
        target_model_factory: Optional[Callable[[QPS], Model]],
        model_pack: Optional[object],
        parameterization: str,
    ) -> Model:
        """Resolve or construct the initial/target model by precedence.

        Args:
            target_model (Model | None): Explicit model to use if provided.
            target_model_factory (Callable[[QPS], Model] | None): Factory that
                builds a model from `processor_spec` if provided.
            model_pack (object | None): Model pack providing
                `target_model(parameterization)` if provided.
            parameterization (str): Parameterization for the model pack
                (e.g., `'TP'`, `'CPTP'`, `'full'`).

        Returns:
            Model: A pyGSTi model compatible with the experiment design.

        """
        if target_model is not None:
            return target_model
        elif target_model_factory is not None:
            return target_model_factory(self._experiment_design.processor_spec)
        elif model_pack is not None:
            return model_pack.target_model(parameterization)
        else:
            return pygsti.models.create_explicit_model(
                self._experiment_design.processor_spec
            )

    def _generate_circuits(self):
        """Convert pyGSTi circuits to QASM-2 circuits for backend execution.

        Returns:
            List[qiskit.QuantumCircuit]: Parsed QASM-2 circuits via
                `qiskit.qasm2.loads`.

        Notes:
            If your backend expects a different IR or needs gate remapping,
            adapt upstream before execution.

        """
        qasm_circs = [
            c.convert_to_openqasm()
            for c in self._experiment_design.all_circuits_needing_data
        ]
        return [loads(c) for c in qasm_circs]

    def _write_counts_to_dataset(self) -> None:
        """Write measured counts from QCMet into pyGSTi `dataset.txt`."""
        ds_template = read_dataset(
            self.config["experiment_design_path"] / "data" / "dataset.txt",
            collision_action="aggregate",
            record_zero_counts=True,
            ignore_zero_count_lines=False,
        )

        outcomes = [
            format(b, f"0{self.num_qubits}b") for b in range(2**self.num_qubits)
        ]
        new_ds = pygsti.data.DataSet(outcome_labels=outcomes)

        for counts, key in zip(
            self.experiment_data["circuit_measurements"],
            ds_template.keys(),
            strict=True,
        ):
            new_ds[key] = counts

        write_dataset(
            self.config["experiment_design_path"] / "data" / "dataset.txt", new_ds
        )

    def _analyze(self) -> Dict[str, Any]:
        """Run GST analysis and compute gate/SPAM quality metrics.

        Args:
            None

        Returns:
            Dict[str, Any]: A dictionary containing:
                - Per-gate process fidelity.
                - Optional per-gate half diamond norm (if `cvxpy` is available).
                - SPAM fidelity (prep-state and aggregate measurement fidelity).

        """
        self._write_counts_to_dataset()
        data = read_data_from_dir(self.config["experiment_design_path"])

        initial = GSTInitialModel(self._ideal_model)
        gst_protocol = self._build_protocol(initial)

        self.gst_results = gst_protocol.run(
            data, checkpoint_path=self.config["experiment_design_path"]
        )
        self.gst_results.write()

        fitted = self._select_best_estimate_model(self.gst_results)
        basis = self._make_pauli_product_basis()

        results: Dict[str, Any] = {}
        results.update(self._compute_gate_metrics(fitted, basis))
        results.update(self._compute_spam_metrics(fitted, basis))
        return results

    # -------------------------------------------------------------------------
    # Analysis helpers
    # -------------------------------------------------------------------------
    def _build_protocol(self, initial: GSTInitialModel):
        """Construct the GST protocol instance based on `use_standard_gst`.

        Args:
            initial (GSTInitialModel): Initial/target model wrapper for GST.

        Returns:
            pygsti.protocols.Protocol: Either `StandardGST` or core
                `GateSetTomography` with optional customization.

        """
        if self._use_standard_gst:
            return StandardGST(initial)
        return GateSetTomography(
            initial,
            gaugeopt_suite=self._gst_gaugeopt_suite,
            objfn_builders=self._gst_objfn_builders,
            optimizer=self._gst_optimizer,
            badfit_options=self._gst_badfit_options,
            verbosity=self._gst_verbosity,
        )

    def _select_best_estimate_model(self, gst_results) -> Model:
        """Select a fitted model, preferring gauge-optimized results when present.

        Args:
            gst_results: The pyGSTi results object produced by running GST.

        Returns:
            Model: The selected fitted model, chosen by preference order:
                'stdgaugeopt' → 'final iteration estimate' → 'final'
                → first available.

        """
        est_name = next(iter(gst_results.estimates.keys()))
        est = gst_results.estimates[est_name]
        for k in ("stdgaugeopt", "final iteration estimate", "final"):
            if k in est.models:
                return est.models[k]
        return next(iter(est.models.values()))

    def _make_pauli_product_basis(self) -> pygsti.baseobjs.Basis:
        """Create the Pauli-product basis over Hilbert–Schmidt space (dimension 4**n).

        Args:
            None

        Returns:
            pygsti.baseobjs.Basis: The `"pp"` basis appropriate for qubit
                superoperators and state/effect super-kets.

        """
        return pygsti.baseobjs.Basis.cast("pp", 4**self.num_qubits)

    def _compute_gate_metrics(
        self, fitted: Model, basis: pygsti.baseobjs.Basis
    ) -> Dict[str, Any]:
        """Compute per-gate process fidelity and (optionally) half diamond norm.

        Args:
            fitted (Model): Fitted pyGSTi model selected from results.
            basis (pygsti.baseobjs.Basis): Pauli-product basis in Hilbert–Schmidt space.

        Returns:
            Dict[str, Any]: Metrics keyed by gate label:
                - 'Process Fidelity Gate:{gate}'
                - 'Diamond Norm of Gate:{gate}' (if `cvxpy` available)

        """
        gates = [label.to_native() for label in self._ideal_model.operations.keys()]
        results: Dict[str, Any] = {}
        print("Gate execution quality metrics:")

        for gate in gates:
            results[f"Process Fidelity Gate:{gate}"] = float(
                rptbl.entanglement_fidelity(
                    self._ideal_model[gate], fitted[gate], basis
                )
            )
            if find_spec("cvxpy") is not None:
                results[f"Diamond Norm of Gate:{gate}"] = float(
                    rptbl.half_diamond_norm(
                        self._ideal_model[gate], fitted[gate], basis
                    )
                )
            else:
                print("Diamond norm not computed (requires cvxpy)")
        return results

    def _compute_spam_metrics(
        self, fitted: Model, basis: pygsti.baseobjs.Basis
    ) -> Dict[str, Any]:
        """Compute SPAM metrics: prep-state fidelity and aggregate measurement fidelity.

        Args:
            fitted (Model): Fitted pyGSTi model selected from results.
            basis (pygsti.baseobjs.Basis): Pauli-product basis in Hilbert–Schmidt space.

        Returns:
            Dict[str, Any]: A dictionary under key 'SPAM fidelity' containing:
                - 'Fidelity of initial density matrix'
                - 'Measurement Fidelity'

        """
        spam: Dict[str, Any] = {}

        rho0_ideal_vec = self._ideal_model["rho0"].to_dense("HilbertSchmidt")
        rho0_est_vec = fitted["rho0"].to_dense("HilbertSchmidt")
        spam["Fidelity of initial density matrix"] = float(
            rptbl.vec_fidelity(rho0_ideal_vec, rho0_est_vec, basis)
        )

        spam["Measurement Fidelity"] = float(
            1
            - rptbl.povm_entanglement_infidelity(self._ideal_model, fitted, "Mdefault")
        )

        return {"SPAM fidelity": spam}

    def __del__(self):
        """Clean up the temporary directory when `save_path` is None."""
        if getattr(self, "save_enabled", None) is False and hasattr(self, "_temp_dir"):
            self._temp_dir.cleanup()
