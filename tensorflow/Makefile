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

TARGETS += tensorflow

TENSORFLOW_TARGETS = tensorflow-mkl tensorflow1 tensorflow2 tensorflow-hbase tensorflow-keras tensorflow-mongodb

tensorflow: $(TENSORFLOW_TARGETS)

tensorflow-mkl: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/tensorflow-1.15.2-mkl.image tensorflow/tensorflow-1.15.2-mkl

tensorflow1: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/tensorflow-1.15.5-cuda121.image tensorflow/tensorflow-1.15.5-cuda121

tensorflow-hbase: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/tensorflow-1.15.5-cuda121-hbase.image tensorflow/tensorflow-1.15.5-cuda121-hbase

tensorflow-keras: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/tensorflow-1.15.5-cuda121-keras.image tensorflow/tensorflow-1.15.5-cuda121-keras

tensorflow-mongodb: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/tensorflow-1.15.5-cuda121-mongodb.image tensorflow/tensorflow-1.15.5-cuda121-mongodb

tensorflow2: $(IMAGE_DIR)
	singularity build $(IMAGE_DIR)/tensorflow-2.12.0-cuda121.image tensorflow/tensorflow-2.12.0-cuda121