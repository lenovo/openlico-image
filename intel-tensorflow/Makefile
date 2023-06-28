# Copyright 2015-2023 Lenovo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

TARGETS += intel-tensorflow

INTEL_TENSORFLOW_TARGETS = intel-tensorflow1 intel-tensorflow2-cpu

intel-tensorflow: $(INTEL_TENSORFLOW_TARGETS)

intel-tensorflow1: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/intel-tensorflow-1.15.2-cpu.image intel-tensorflow/intel-tensorflow-1.15.2-cpu

intel-tensorflow2-cpu: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/intel-tensorflow-2.10.0-cpu.image intel-tensorflow/intel-tensorflow-2.10.0-cpu
