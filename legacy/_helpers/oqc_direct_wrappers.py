from qcaas_client.client import OQCClient, QPUTask, QPUTaskResult, TaskStatus
from qcaas_client.compiler_config import (
    CompilerConfig,
    QuantumResultsFormat,
    Tket,
    TketOptimizations,
)

# Add credentials to access OQC QCAAS Platform
QCAAS_URL = "https://qcaas.oqc.app"
QCAAS_EMAIL = ""
QCAAS_PASSWORD = ""


class OQCTaskResultWrapper:
    def __init__(self, result: QPUTaskResult) -> None:
        self.measurement_counts = result.result["c"]


class OQCTaskWrapper:
    def __init__(self, task: QPUTask, client: OQCClient) -> None:
        self.task = task
        self.id = task.task_id
        self.client = client

    def result(self):
        return OQCTaskResultWrapper(self.client.get_task_results(self.id))

    def state(self):
        task_status = self.client.get_task_status(self.id)
        return "COMPLETED" if task_status == TaskStatus.COMPLETED else "NOT COMPLETED"


class OQCTaskBatchWrapper:
    def __init__(self, tasks: list[OQCTaskWrapper]) -> None:
        self.tasks = tasks


class OQCClientWrapper:
    def __init__(self) -> None:
        self.client = OQCClient(
            url=QCAAS_URL, email=QCAAS_EMAIL, password=QCAAS_PASSWORD
        )
        self.client.authenticate()
        self.res_format = QuantumResultsFormat().binary_count()
        self.optimisations = Tket()
        # TketOptimizations.DefaultMappingPass.disable()
        self.optimisations.tket_optimizations = TketOptimizations.One

    def run_batch(self, circuits: list[str], shots=1000, max_parallel=100, **kwargs):
        config = CompilerConfig(
            results_format=self.res_format,
            repeats=shots,
            optimizations=self.optimisations,
        )
        tasks = [QPUTask(qasm, config) for qasm in circuits]
        tasks = OQCTaskBatchWrapper(
            [OQCTaskWrapper(t, self.client) for t in self.client.schedule_tasks(tasks)]
        )

        return tasks
