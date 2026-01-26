
## The Circuit Submitter

1. [Overview](#1-overview)
1. [Replacing the circuit submitter](#2-replacing-the-circuit-submitter)
2. [AWS Braket](#3-aws-braket)
3. [Creating a new backend for the CircuitSubmitter
](#4-creating-a-new-backend-for-the-circuitsubmitter)

###  1. Overview

This readme file gives an overview on how to modify the CircuitSubmitter to add your own backend, or how to use it to run on AWS Braket. At the time of writing, benchmarks were run on quantum computers available on Braket, many of which are now unavailable. Nevertheless, the software integration of Braket still exists in the circuit sumbitter class, and it may be used to extend the software to other quantum computers available on Braket or even other hardware backends.

###  2. Replacing the circuit submitter

If you would like to use different quantum hardware, you may create a new device backend. Please follow the existing backends as a guide to implement your own device. Alternatively, the metric software has been designed so that circuit submission software is not heavily integrated with it. As a result, it will be straight forward to replace it with an interface of your choice. One must provide an interface that accepts circuits of the class `qiskit.circuit.QuantumCircuit`, or accepts QASM strings or paths to QASM files, and then provides a list of counts for each circuit in the same format that qiskit produces. 



###  3. AWS Braket

At the time of writing the report, we ran selected benchmarks using AWS Braket. In order to set up your computer for running on AWS Braket, you must first create an account with [AWS and ensure that all necessary requirements are in place to submit circuits to their hardware](https://aws.amazon.com/braket/).

Once you have an account, to run via **AWS Braket**, you need to create a directory `.aws` in your home directory, and create two files under it:
 - a file named `config`, containing the following text:
    ```
    [default]
    region = us-east-1
    ```
 - a file named `credentials`, containing the following text:
    ```
    [default]
    aws_access_key_id = <aws_access_key_id>
    aws_secret_access_key = <aws_secret_access_key>
    ```
    where you should replace `<aws_access_key_id>` and `<aws_secret_access_key>` with your AWS Access Key ID and AWS Secret Access Key respectively, without the triangular brackets.

The folder structure should then be of the following form:

| Operating system | Default location and name of files  |
|------------------|-------------------------------------|
| Linux and macOS  | `~/.aws/config`                      |
|                  | `~/.aws/credentials`                  |
| Windows          | `%USERPROFILE%\.aws\config`           |
|                  | `%USERPROFILE%\.aws\credentials`      |


Note that many of the devices that were used during the creation of this software are now unavailable. However, the software included in `_helpers\circuit_submitter.py`, and `_helpers\backend_helpers.py` can be used as a starting point for the user to adjust the software as necessary to run on new quantum computers available on AWS Braket.


###  4. Creating a new backend for the CircuitSubmitter

To create a new backend using the `CircuitSubmitter()` class, first the user must create client, task, task batch, and task result wrappers. An example on how to create the wrappers can be seen in the file `_helpers/noisy_simulator_wrappers.py` and `_helpers/backend_helpers.py`

Specifically, if you want to create a new backend, then you must create the following classes, and modify one function as listed in the following table:


| Type of object | Name | Path to modify | Notes |
|---|---|---|---|
| class | MyQPU(AwsBackendHelper) | `_helpers\backend_helpers.py` | imported by `_helpers\circuit_submitter.py` |
| existing function to modify | get_backend_helper(name) | `_helpers\backend_helpers.py` | imported by `_helpers\circuit_submitter.py` |
| class | ClientWrapper | `_helpers\noisy_simulator_wrappers.py` | imported by `_helpers\backend_helpers.py`. Can create in separate file but ensure to import class in `backend_helpers.py` |
| class | TaskBatchWrapper | `_helpers\noisy_simulator_wrappers.py` | imported by `_helpers\backend_helpers.py`. Can create in separate file but ensure to import class in `backend_helpers.py` |
| class | TaskWrapper | `_helpers\noisy_simulator_wrappers.py` | imported by `_helpers\backend_helpers.py`. Can create in separate file but ensure to import class in `backend_helpers.py` |
| class | TaskResultWrapper | `_helpers\noisy_simulator_wrappers.py` | imported by `_helpers\backend_helpers.py`. Can create in separate file but ensure to import class in `backend_helpers.py` |
|  |  |  |  |

Additionally, you may need to modify the CircuitSubmitter class in the file `_helpers\circuit_submitter.py`

Within the CircuitSubmitter class, we have the following function which may need modifying:

```python
class CircuitSubmitter()
    ...
    ...
    ...
    
    def submit_circuits(self, shots: int, verbatim: bool = True, skip_asking: bool = False, 
                        skip_transpilation: bool = False, print_summary: bool = True,
                        braket_circuits: list[Circuit] = None, qasm_strs: list[str] = None, 
                        qasm_paths: list[str] = None,  inputs: dict[str, float] = None) -> Union[list[AwsQuantumTask], list[LocalQuantumTask]]:

```

Specifically, if your new backend has native gates such that circuits would require transpilation prior to execution, then you must add an `elif` statement in `submit_circuits()` function after line 150 for your device name and then add corresponding code to transpile the circuits to the native gates of your backend, ensuring that the transpiled circuits are assigned to the variable `circuits`

When implementing the CircuitSubmitter, we sucessfully ran benchmarks on the AWS Braket hardware devices that were available at the time of writing. Whilst they may now be decomissioned, the backend for the 'Lucy' device may serve as a useful starting point for the user to add their own backend. 

In addition, we also include here some basic descriptions of the classes that must be added, and the functions that must be modified below.


```Python
class TaskResultWrapper:
    def __init__(self, result: QPUTaskResult) -> None:
        '''Code to get measurement counts for task
        '''


class TaskWrapper:
    def __init__(self, task: QPUTask, client: Client) -> None:
        self.task = task
        ...

    def result(self):
        '''Code to get task result
        '''

    def state(self):
        '''Code to get task status. Either return "COMPLETED" or "NOT COMPLETED"
        '''


class TaskBatchWrapper:
    def __init__(self, tasks: list[TaskWrapper]) -> None:
        self.tasks = tasks


class ClientWrapper:
    def __init__(self) -> None:
        '''Code to initialize your backend here
        '''


    def run_batch(self, circuits: list[str], shots=1000, max_parallel=100, **kwargs):
        '''Code to send a batch of circuits to backend to execute. Should return a list of TaskWrappers which are then used to return a TaskBatchWrapper 
        '''

```


In the file `_helpers/backend_helpers.py` add the following for your hardware/emulator:

Create a class that inherits from this class and create the methods listed in this abstract base class.

```python
class AwsBackendHelper(ABC):
    def __init__(self) -> None:
        self.device = None

    def get_device_calibration(self) -> str:
        return self.device.properties.json()

    @abstractmethod
    def get_qiskit_backend(self) -> Union[BraketLocalBackend, AWSBraketBackend]:
        pass

    @abstractmethod
    def get_basis_gates(self) -> list[str]:
        pass

    @abstractmethod
    def get_costs(self) -> Tuple[float]:
        """Get cost_per_circuit and cost_per_shot"""
        pass
```
You would, for example, create a class called `class MyQPU(AwsBackendHelper):` and also write the associated methods. This file would need to import all of the wrappers you have written for your backend. 

Then, you will need to add the name of this new class to the function at the bottom of the file to be able to call it from the circuit submitter, for example:

```python
def get_backend_helper(name: str) -> AwsBackendHelper:
    if name == "simulator":
        return IdealSimulatorHelper()
    ...
    ...
    elif name == "MyQPU":
        return MyQPU()
```


