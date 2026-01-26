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
"""
Please note that the software in this file is intended to be used as an example for 
how one may run T1 using pulse level access. This software was used during the time
of writing https://doi.org/10.48550/arXiv.2502.06717 and the devices it measured have 
now been decomissioned.
"""
import numpy as np
import sys
import os
import json
from time import strftime, gmtime
from datetime import datetime
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.circuits import gates
from braket.circuits import Circuit
from braket.pulse import PulseSequence
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")
font_size = 12
plt.rcParams.update({'font.size': font_size})


def sf(x, digits=4):
    return np.format_float_positional(x, precision=digits, unique=False, fractional=False, trim='k')


def exp_func(x, amp, dr):
    return amp * np.exp(x * -1 / dr)

SIM = 0
RIGETTI = 1
OQC = 2
IONQ = 3

device_choice = OQC

device_names = ["simulator", "Rigetti Aspen-M-3", "OQC Lucy", "IonQ Harmony"]
device_name = device_names[device_choice]
device_ids = ["simulator",
              "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3",
              "arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
              "arn:aws:braket:us-east-1::device/qpu/ionq/Harmony"]
device_id = device_ids[device_choice]

SCRIPT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
RESULTS_PATH = "results"
os.makedirs(RESULTS_PATH, exist_ok=True)
TIME_PATH = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
PATH = os.path.join(SCRIPT_PATH, RESULTS_PATH, TIME_PATH)
os.makedirs(PATH, exist_ok=True)

if device_choice != SIM:
    device = AwsDevice(device_id)
else:
    device = LocalSimulator("braket_dm")

qubit = 4  # Rigetti: low to high: 144, 130, 103, 34, 21, 7, 5, 15, 17

time = f"{device_name} Run on {datetime.today().strftime('%Y-%m-%d %H.%M.%S')} (Pulse) Qubit {qubit}"
stdout, stderr = sys.stdout, sys.stderr
sys.stdout = open(PATH + f"/{time}.txt", 'w')
# sys.stderr = sys.stdout


max_num_identities = 100
shots = 500
wait_time_s = 1e-6
if device_choice == RIGETTI:
    wait_pulse = PulseSequence().delay(device.frames[f"q{qubit}_rf_frame"], wait_time_s) # FOR RIGETTI
else:
    wait_pulse = PulseSequence().delay(device.frames[f"q{qubit}_drive"], wait_time_s)
# Loop over circuit length
if device_choice == RIGETTI:
    reported_t1 = float(device.properties.provider.specs['1Q'][str(qubit)]['T1']) * 1e6
    print(f'{reported_t1=}')
    max_t = reported_t1 * 3.5
    max_steps = 10
    scaling_factor = max_steps/2
    # Run from t_min = 0 to max_t = (3.5 * reported_t1)
    nums_identities = np.unique(np.cumsum(np.around(np.linspace(0, max_t / scaling_factor, max_steps))).astype(int)).tolist()
    
elif device_choice == IONQ:
    nums_identities = np.array([0, 10, 20, 30, 40, 50])
elif device_choice == OQC:
    reported_t1 = float(device.properties.provider.properties["one_qubit"][str(qubit)]['T1'])
    print(f'{reported_t1=}')
    max_t = reported_t1 * 1.2
    max_steps = 14
    scaling_factor = max_steps/2
    # Run from t_min = 0 to max_t = (3.5 * reported_t1)
    nums_identities = np.unique(np.cumsum(np.around(np.linspace(0, max_t / scaling_factor, max_steps))).astype(int)).tolist()
    
else:
    nums_identities = np.arange(0, max_num_identities + 1, 4)

print(f'{nums_identities=}')

circuit_list = []
for num_identities in nums_identities:
    t1_circuit = Circuit().rx(qubit, np.pi) if device_choice == RIGETTI else Circuit().x(qubit)
    # Add idle gates
    for j in range(num_identities):
        t1_circuit = t1_circuit.pulse_gate(qubit, wait_pulse)

    print(t1_circuit)

    # Verbatim compilation - make sure circuit is not optimised
    t1_circuit_verbatim = Circuit().add_verbatim_box(t1_circuit)
    # print(t1_circuit_verbatim)
    circuit_list.append(t1_circuit_verbatim)
    
print(len(circuit_list))
try:
    print("Submit job")
    results = device.run_batch(circuit_list, max_parallel=100 ,shots=shots).results()
    # result = device.run(circuit_list[0], shots=10).result()
    # print(result, file=stderr)
    # exit(1)
    
except AttributeError as e:
    print(f"\nPulse Run with {num_identities} failed due to circuit too long\n")
    raise e


probabilities = []
result_probs = []

for result in results:
    result_prob  = result.measurement_probabilities
    result_probs.append(result_prob)
    if '1' in result_prob.keys():
        p1 = result_prob['1']
    else:
        p1 = 1 - result_prob['0']
    probabilities.append(p1)
print(result_probs)

print(f"\nPulse Run with all waits finished in {datetime.today().strftime('%Y-%m-%d %H.%M.%S')}\n")


nums_identities = nums_identities[:len(probabilities)]
# Fit for decay parameters
popt, pcov = curve_fit(exp_func, nums_identities, probabilities, p0=[1, 1000], method='trf')
std = np.sqrt(np.diag(pcov))
fitted_probs = exp_func(np.array(nums_identities), *popt)
fitted_probs = np.around(fitted_probs, decimals=4)
print(f"Hardware probabilities of 1 state: {probabilities}\n"
      f"Fitted probabilities: {fitted_probs.tolist()}\n"
      f"Fitted covariance matrix: {pcov.tolist()}, std: {std.tolist()}\n"
      f"Fitted amplitude and decay rate (in units of {wait_time_s}s): {popt.tolist()}\n"
      # f"AWS Reported T1 time: {device.properties.provider.properties['one_qubit'][str(qubit)]['T1']}")
      f"AWS Reported T1 time: {reported_t1}")

# Save diagram
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(nums_identities, probabilities, label="Hardware results")
# r"$" + np.around(popt[0], 4) + r"*e^{\frac{-x}{" + np.around(popt[1], 4) + r"}}"
ax.plot(nums_identities, fitted_probs, label=f"Fitted equation:" + r"$a*exp[-x/T_1]$" + f"\nT_1: {sf(popt[1])}" + r"$\mu$s")
ax.set_xlabel(f"Idle time " + r"($\mu$s)")
ax.set_xticks(nums_identities)
ax.set_ylim(0, 1)
ax.set_ylabel(r"$|⟨1|\psi⟩|^2$")
ax.legend()
ax.text(x=0, y=0.01, s=f"{time}\nFitted amplitude: {sf(popt[0])}, std: {sf(std[0])}\n"
         f"Fitted T_1: {sf(popt[1])}" + r"$\mu$s" + f", std: {sf(std[1])}" + r"$\mu$s" + f"\n"
         f"Reported T_1: {sf(reported_t1)}" + r"$\mu$s", size=font_size-4)
plt.title(f"T_1 time experiment on {device_name} device on qubit {qubit}")
plt.savefig(PATH + f"/{device_name} Q{qubit}.png", format="png")
# plt.show()
plt.clf()

sys.stdout.close()
sys.stdout, sys.stderr = stdout, stderr
