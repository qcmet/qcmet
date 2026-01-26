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

# Load in data
data = pygsti.io.read_data_from_dir(gst_folder)

# Perform CPTP-constrained GST
protocol = pygsti.protocols.StandardGST("CPTPLND",  verbosity=3)
protocol.optimizer.maxiter = 1000
start = time.time()
results = protocol.run(data)
end = time.time()
print(f"Finished in {end - start:.1f}s")


results.write()
results = pygsti.io.read_results_from_dir(gst_folder, name="StandardGST")
report = pygsti.report.construct_standard_report(results, title="GST", verbosity=2)
report.write_html(gst_folder, verbosity=2)
