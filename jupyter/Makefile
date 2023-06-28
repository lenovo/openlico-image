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

TARGETS += jupyter

JUPYTER_TARGETS = jupyter-py37 jupyter-py38 jupyter-intel-py37 jupyter-intel-optimized-pytorch jupyter-intel-optimized-tensorflow jupyter-default

jupyter: $(JUPYTER_TARGETS)

jupyter-py37: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/jupyter-py37.image jupyter/jupyter-py37

jupyter-py38: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/jupyter-py38.image jupyter/jupyter-py38

jupyter-default: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/jupyter-default.image jupyter/jupyter-default

jupyter-intel-py37: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/jupyter-intel-py37.image jupyter/jupyter-intel-py37

jupyter-intel-optimized-pytorch: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/jupyter-intel-optimized-pytorch.image jupyter/jupyter-intel-optimized-pytorch

jupyter-intel-optimized-tensorflow: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/jupyter-intel-optimized-tensorflow.image jupyter/jupyter-intel-optimized-tensorflow