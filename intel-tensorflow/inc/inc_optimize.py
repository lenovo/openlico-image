import os
import cv2
import shutil
import sys
import json
import zipfile
import numpy as np
from datetime import datetime
import tensorflow.compat.v1 as tf
from neural_compressor.experimental import (Graph_Optimization, Quantization,
                                            common)
from neural_compressor.experimental.data.datasets.dataset import \
    TensorflowImageFolder
from neural_compressor.experimental.data.transforms import BaseTransform
from inc_common import get_pb_node_shape


class MyResizeTFTransform(BaseTransform):
    """Resize the input image to the given size.

    Args:
        size (list or int): Size of the result
        interpolation (str, default='bilinear'):
            Desired interpolation type,support 'bilinear', 'nearest', 'bicubic'

    Returns:
        tuple of processed image and label
    """

    def __init__(self, size, interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]
        self.interpolation = interpolation

        if self.interpolation not in ['bilinear', 'nearest', 'bicubic']:
            raise ValueError('Unsupported interpolation type!')

    def __call__(self, sample):
        image, label = sample
        if isinstance(image, tf.Tensor):
            image = tf.image.resize(image, self.size,
                                    method=self.interpolation)
        else:
            image = cv2.resize(image, self.size)
        np_img = self.per_image_standardization(image)
        np_img = np_img[:, :, [2, 1, 0]]
        return (np_img, label)

    def per_image_standardization(self, img):
        mean = np.mean(img)
        std = np.std(img)
        count = 1
        for size in img.shape:
            count = count * size
        adj_std = max(std, 1.0 / np.sqrt(count))
        img = (img - mean) / adj_std
        return img


class MyMetric(object):
    def __init__(self, *args):
        self.pred_list = []
        self.label_list = []
        self.samples = 0
        pass

    def update(self, predict, label):
        self.pred_list.extend(np.argmax(predict, axis=1))
        self.label_list.extend(label)
        self.samples += len(label)
        pass

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0
        pass

    def result(self):
        correct_num = np.sum(
            np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.samples


class NeuralCompressor(object):
    def __init__(self, image_size, input_model, output_path, batch_size=8,
                 dummy_dataset=False):
        self.dummy_dataset = dummy_dataset
        self.image_size = image_size
        self.input_model = input_model
        self.output_path = output_path
        self.batch_size = batch_size
        self.output_model = output_path + '/inc_optimized_model.pb'
        self.config_path = "/opt/inc/conf.yaml"

    def graphoptimization(self, target_mode):
        if target_mode == "inc_bf16":
            precisions = "bf16,fp32"
        else:
            precisions = "fp32"
        graph_optimizer = Graph_Optimization()
        graph_optimizer.precisions = precisions
        graph_optimizer.model = common.Model(self.input_model)
        optimized_model = graph_optimizer()
        if optimized_model is not None:
            optimized_model.save(self.output_model)
        else:
            print("[ERROR]:\n"
                  "Specified timeout or max trials is reached!"
                  "Not found any quantized model which meet accuracy goal."
                  "Exit.")
            sys.exit(100)

    def quantizeByDummy(self):
        quantizer = Quantization(self.config_path)
        dataset = quantizer.dataset('dummy_v2', input_shape=(
            self.image_size, self.image_size, 3), label_shape=(1,))
        quantizer.model = common.Model(self.input_model)
        quantizer.calib_dataloader = common.DataLoader(
            dataset, batch_size=self.batch_size)
        quantized_model = quantizer.fit()
        if quantized_model is not None:
            quantized_model.save(self.output_model)
        else:
            print("[ERROR]:\n"
                  "Specified timeout or max trials is reached!"
                  "Not found any quantized model which meet accuracy goal."
                  "Exit.")
            sys.exit(100)

    def quantize(self, dataset_path):
        quantizer = Quantization(self.config_path)
        transform = MyResizeTFTransform([self.image_size, self.image_size],
                                        interpolation='bilinear')
        dataset = TensorflowImageFolder(root=dataset_path, transform=transform)
        quantizer.model = common.Model(self.input_model)
        quantizer.eval_dataloader = common.DataLoader(
            dataset, batch_size=self.batch_size)
        quantizer.calib_dataloader = common.DataLoader(
            dataset, batch_size=self.batch_size)
        quantizer.metric = common.Metric(MyMetric, 'metric_name')
        quantized_model = quantizer.fit()
        if quantized_model is not None:
            quantized_model.save(self.output_model)
        else:
            print("[ERROR]:\n"
                  "Specified timeout or max trials is reached!"
                  "Not found any quantized model which meet accuracy goal."
                  "Exit.")
            sys.exit(100)


class Optimization(object):

    def __init__(self, frozen_pb_path, model_name, output_path,
                 target_mode='inc_int8', dummy_dataset=False,
                 calibration_batch_size=8, calibration_dataset_path='',
                 use_server=False):
        self.input_node_name = "lico_input_node"
        self.output_node_name = "lico_output_node"
        self.frozen_pb_path = frozen_pb_path
        self.model_name = model_name
        self.output_path = output_path
        self.target_mode = target_mode
        self.dummy_dataset = dummy_dataset
        self.calibration_batch_size = calibration_batch_size
        self.calibration_dataset_path = calibration_dataset_path
        self.use_server = use_server
        self.tmp_dir = os.path.join(self.output_path, 'tmp')

    def packaging_files(self):
        print("[INFO] Start packaging export files.", flush=True)

        # Copy the inc folder to tmp_dir
        inc_script_dir = os.path.join(self.tmp_dir, 'inc')
        if os.path.exists(inc_script_dir):
            shutil.rmtree(inc_script_dir)
        shutil.copytree('/opt/inc', inc_script_dir)

        # Verify that the file is generated
        for file_name in ['labels.txt',
                          'inc_optimized_model.pb',
                          self.model_name + '_frozen_model.pb']:
            this_file = os.path.join(self.tmp_dir, file_name)
            if not os.path.exists(this_file):
                print(f"[ERROR] The export file is not fully generated. "
                      f"Missing {file_name} file.\n"
                      f"Packaging to terminate...\n"
                      f"Clearing Residual Files...")
                shutil.rmtree(self.tmp_dir)
                return

        # Write model_name in file
        with open(os.path.join(self.tmp_dir, 'model_define.json'),
                  'w', encoding='UTF-8') as f:
            model_define = {
                'model_name': self.model_name,
                'category': 'INC'
            }
            f.write(json.dumps(model_define))

        # Make zip file
        zip_name = f'model_{self.target_mode}_' \
                   f'{int(datetime.now().timestamp())}.zip'
        with zipfile.ZipFile(
                os.path.join(self.output_path, zip_name),
                'w',
                zipfile.ZIP_DEFLATED
        ) as zf:
            for path, dirnames, filenames in os.walk(self.tmp_dir):
                for filename in filenames:
                    zf.write(
                        os.path.join(path, filename),
                        os.path.join(path.replace(self.tmp_dir, ''),
                                     filename)
                    )
        shutil.rmtree(self.tmp_dir)
        print(f"[INFO] Finished. You can download files from "
              f"{os.path.join(self.output_path, zip_name)}")

    def optimize(self):
        self.image_size = int(
            get_pb_node_shape(self.frozen_pb_path, self.input_node_name)[1])
        print(f"[INFO] Optimized frozen pb with mode: {self.target_mode}")
        if self.target_mode.lower() in ["inc_fp32", "inc_bf16"]:
            nc = NeuralCompressor(self.image_size, self.frozen_pb_path,
                                  self.tmp_dir)
            nc.graphoptimization(self.target_mode.lower())
        else:
            nc = NeuralCompressor(self.image_size, self.frozen_pb_path,
                                  self.tmp_dir,
                                  self.calibration_batch_size,
                                  dummy_dataset=self.dummy_dataset)
            if self.dummy_dataset:
                print("[INFO] Optimized with dummy dataset")
                nc.quantizeByDummy()
            else:
                print("[INFO] Optimized with calibration dataset")
                nc.quantize(self.calibration_dataset_path)
        self.packaging_files()


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('frozen_pb_path', '', 'path of frozen model')
    tf.app.flags.DEFINE_string('model_name', '', '')
    tf.app.flags.DEFINE_string('output_path', '.',
                               'store path of output model')
    tf.app.flags.DEFINE_string('device', 'GPU', '')
    tf.app.flags.DEFINE_string('target_mode', 'float32', '')
    tf.app.flags.DEFINE_bool('dummy_dataset', False, 'dummy dataset')
    tf.app.flags.DEFINE_string('calib_ds', '', 'calibration dataset')
    tf.app.flags.DEFINE_integer('calibration_batch_size', 8,
                                'dataset batch size')
    tf.app.flags.DEFINE_bool('use_server', False,
                             'whether to add server files')

    FLAGS = tf.app.flags.FLAGS

    print(f"[INFO] Params: \n"
          f"frozen_pb_path: {FLAGS.frozen_pb_path} \n"
          f"model_name: {FLAGS.model_name} \n"
          f"output_path: {FLAGS.output_path} \n"
          f"device: {FLAGS.device} \n"
          f"target_mode: {FLAGS.target_mode} \n"
          f"dummy_dataset: {FLAGS.dummy_dataset} \n"
          f"calibration_dataset_path: {FLAGS.calib_ds} \n"
          f"calibration_batch_size: {FLAGS.calibration_batch_size} \n"
          f"use_server: {FLAGS.use_server}", flush=True)

    opt = Optimization(FLAGS.frozen_pb_path, FLAGS.model_name,
                       FLAGS.output_path, FLAGS.target_mode,
                       FLAGS.dummy_dataset, FLAGS.calibration_batch_size,
                       FLAGS.calib_ds)
    opt.optimize()
