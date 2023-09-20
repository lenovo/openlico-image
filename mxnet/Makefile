# Copyright 2015-present Lenovo
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

TARGETS += mxnet

MXNET_TARGETS = mxnet-cpu mxnet-gpu

mxnet: $(MXNET_TARGETS)

mxnet-cpu: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/mxnet-1.9.1-cpu.image mxnet/mxnet-1.9.1-cpu

mxnet-gpu: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/mxnet-1.9.1-gpu-cuda112.image mxnet/mxnet-1.9.1-gpu-cuda112