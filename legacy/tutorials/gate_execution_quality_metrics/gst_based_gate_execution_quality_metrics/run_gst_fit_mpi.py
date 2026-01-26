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

import time
import pygsti
import pathlib


# Specify the path to the gst folder
gst_folder = pathlib.Path('gst_data')

# Get MPI comm
from mpi4py import MPI
# If you do not use MPI, use non MPI script
comm = MPI.COMM_WORLD

print("Rank %d started" % comm.Get_rank())

data = pygsti.io.read_data_from_dir(gst_folder, comm=comm)
memLim = 9*(1024)**3  # 9 GB

protocol = pygsti.protocols.StandardGST("CPTPLND",  verbosity=3)
protocol.optimizer.maxiter = 1000
start = time.time()
results = protocol.run(data, memlimit=memLim, comm=comm)
end = time.time()
if comm is not None:
    print("Rank %d finished in %.1fs" % (comm.Get_rank(), end - start))
else:
    print(f"Finished in {end - start:.1f}s")

if comm is not None:
    if comm.Get_rank() == 0:
      results.write() 
    results=None 
else:
    results.write()

results = pygsti.io.read_results_from_dir(gst_folder, name="StandardGST", comm=comm)
report = pygsti.report.construct_standard_report(results, title="GST", verbosity=2, comm=comm)
report.write_html(gst_folder, verbosity=2)
